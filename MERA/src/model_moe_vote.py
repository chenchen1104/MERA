# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import math
import json
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from qlib.model.base import Model

from src.custom_moe_layer import FMoETransformerMLP
from src.noisy_gate import NoisyGate
from src.noisy_gate_vmoe import NoisyGate_VMoE

from src.dataset import collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


class TRAModel(Model):
    def __init__(
        self,
        model_config,
        tra_config,
        model_type="LSTM",
        lr=1e-3,
        n_epochs=500,
        early_stop=50,
        smooth_steps=5,
        max_steps_per_epoch=None,
        freeze_model=False,
        model_init_state=None,
        lamb=0.0,
        rho=0.99,
        seed=None,
        logdir=None,
        eval_train=True,
        eval_test=True,
        avg_params=True,
        **kwargs,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.logger = get_module_logger("TRA")
        self.logger.info("TRA Model...")

        self.model = eval(model_type)(**model_config).to(device)
        if model_init_state:
            self.model.load_state_dict(torch.load(model_init_state, map_location="cuda")["model"],strict=False)
        if freeze_model:
            # for param in self.model.parameters():
            #     param.requires_grad_(False)
            for name, param in self.model.named_parameters():
                if 'experts' in name or 'gate' in name or 'bn' in name or 'norm' in name or 'input_proj' in name or 'embedding' in name:
                    print(name)
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        else:
            self.logger.info("# model params: %d" % sum([p.numel() for p in self.model.parameters()]))

        self.tra = TRA(self.model.output_size, **tra_config).to(device)
        self.logger.info("# tra params: %d" % sum([p.numel() for p in self.tra.parameters()]))

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=lr)

        self.model_config = model_config
        self.tra_config = tra_config
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        self.rho = rho
        self.seed = seed
        self.logdir = logdir
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.avg_params = avg_params

        if self.tra.num_states > 1 and not self.eval_train:
            self.logger.warn("`eval_train` will be ignored when using TRA")

        if self.logdir is not None:
            if os.path.exists(self.logdir):
                self.logger.warn(f"logdir {self.logdir} is not empty")
            os.makedirs(self.logdir, exist_ok=True)

        self.fitted = False
        self.global_step = -1

    def collect_noisy_gating_loss(self, model, weight):
        loss = 0
        for module in model.modules():
            # print(module)
            if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_loss:
                loss += module.get_loss()
        return loss * weight
    
    def train_epoch(self, data_set):
        self.model.train()
        self.tra.train()

        count = 0
        total_loss = 0
        total_count = 0

        for i, batch in enumerate(data_set):

            self.global_step += 1

            feature, similar, label_raw, label_norm, index = batch

            feature = feature.to(device)
            similar = similar.to(device)
            label_raw = label_raw.to(device)
            label_norm = label_norm.to(device)

            feature = feature.permute(0, 2, 1)
            hidden = self.model(feature, similar)
            pred, all_preds, prob = self.tra(hidden)

            mask = ~torch.isnan(label_norm)

            # 过滤掉NaN值，只保留非NaN值的预测值和标签值
            filtered_pred = pred[mask]
            filtered_label_norm = label_norm[mask]

            loss = (filtered_pred - filtered_label_norm).pow(2).mean()
            # loss = (pred - label_norm).pow(2).mean()

            gate_loss = self.collect_noisy_gating_loss(self.model, 0.01)
            loss += gate_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_count += len(pred)

        total_loss /= total_count
        print(total_loss)
        return total_loss

    def test_epoch(self, data_set, return_pred=False):
        self.model.eval()
        self.tra.eval()

        preds = []
        metrics = []
        for i, batch in enumerate(data_set):
            # print(i)
            feature, similar, label_raw, label_norm, index = batch

            feature = feature.to(device)
            similar=similar.to(device)
            label_raw = label_raw.to(device)
            label_norm = label_norm.to(device)
            
            feature = feature.permute(0, 2, 1)

            with torch.no_grad():
                hidden = self.model(feature, similar)
                pred, all_preds, prob = self.tra(hidden)
            X = np.c_[
                pred.cpu().numpy(),
                label_raw.cpu().numpy(),
            ]

            columns = ["score", "label"]
            pred = pd.DataFrame(X, index = index, columns = columns)

            metrics.append(evaluate(pred))

            if return_pred:
                preds.append(pred)

        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.sort_index(inplace=True)

        return metrics, preds

    def fit(self, train_dataset, val_dataset, test_dataset, evals_result=dict()):

        train_loader = DataLoader(train_dataset, batch_size = 2, collate_fn = collate_fn, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 1, collate_fn = collate_fn, shuffle = False)
        test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = collate_fn, shuffle = False)

        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        params_list = {
            "model": collections.deque(maxlen=self.smooth_steps),
            "tra": collections.deque(maxlen=self.smooth_steps),
        }
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        # train
        self.fitted = True
        self.global_step = -1

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(train_loader)

            self.logger.info("evaluating...")
            # average params for inference
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            self.model.load_state_dict(average_params(params_list["model"]))
            self.tra.load_state_dict(average_params(params_list["tra"]))

            # NOTE: during evaluating, the whole memory will be refreshed
            if self.tra.num_states > 1 or self.eval_train:
                train_metrics = self.test_epoch(train_loader)[0]
                evals_result["train"].append(train_metrics)
                self.logger.info("\ttrain metrics: %s" % train_metrics)

            valid_metrics = self.test_epoch(val_loader)[0]
            evals_result["valid"].append(valid_metrics)
            self.logger.info("\tvalid metrics: %s" % valid_metrics)

            if self.eval_test:
                test_metrics = self.test_epoch(test_loader)[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("\ttest metrics: %s" % test_metrics)
                
            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

            # restore parameters
            self.model.load_state_dict(params_list["model"][-1])
            self.tra.load_state_dict(params_list["tra"][-1])

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])

        metrics, preds = self.test_epoch(test_loader, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)

        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat({name: pd.DataFrame(evals_result[name]) for name in evals_result}, axis=1).to_csv(
                self.logdir + "/logs.csv", index=False
            )

            torch.save(best_params, self.logdir + "/model.bin")

            preds.to_pickle(self.logdir + "/pred.pkl")

            info = {
                "config": {
                    "model_config": self.model_config,
                    "tra_config": self.tra_config,
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "early_stop": self.early_stop,
                    "smooth_steps": self.smooth_steps,
                    "lamb": self.lamb,
                    "rho": self.rho,
                    "seed": self.seed,
                    "logdir": self.logdir,
                },
                "best_eval_metric": -best_score,  # NOTE: minux -1 for minimize
                "metric": metrics,
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, test_dataset, model_path):
        test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = collate_fn, shuffle = False)
        best_params = torch.load(model_path)
        self.model.load_state_dict(best_params["model"])
        self.tra.load_state_dict(best_params["tra"])
        metrics, preds = self.test_epoch(test_loader, return_pred=True)
        print(self.model.moe.score)
        self.logger.info("test metrics: %s" % metrics)
        return preds


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.pe = self.pe.to(device)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(
        self,
        input_size=16,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        input_drop=0.0,
        noise_level=0.0,
        topk=1,
        num_expert=4,
        gate_dim=16,
        moe_gate_type='noisy_vmoe',
        vmoe_noisy_std=1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.noise_level = noise_level
        self.gate_dim=gate_dim

        # self.input_drop = nn.Dropout(input_drop)

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(hidden_size, dropout)

        if moe_gate_type == "noisy":
            moe_gate_fun = NoisyGate
        elif moe_gate_type == "noisy_vmoe":
            moe_gate_fun = NoisyGate_VMoE
        else:
            raise ValueError("unknow gate type of {}".format(moe_gate_type))

        act_layer=nn.GELU
        activation = nn.Sequential(
                act_layer(),
                nn.Dropout(dropout)
            )
        
        blocks = []
        for i in range(num_layers):
            blocks.append(nn.TransformerEncoderLayer(nhead = num_heads, dropout = dropout, d_model = hidden_size, dim_feedforward = hidden_size * 4, norm_first=True))
        
        self.moe = FMoETransformerMLP(num_expert = num_expert, d_model = hidden_size, d_gate = gate_dim, gate = moe_gate_fun,
                                        top_k = topk, activation = activation, vmoe_noisy_std=vmoe_noisy_std, expert_prune=False)
        
        self.encoder = nn.Sequential(*blocks)

        self.output_size = hidden_size
        self.bn = nn.BatchNorm1d(input_size)

        self.embedding = nn.Embedding(10, gate_dim)

    def forward(self, x, similars):
        shape = x.shape
        x = x.reshape(-1,self.input_size)
        x = self.bn(x)
        x = x.reshape(shape)

        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be sequence

        x = self.input_proj(x)
        x = self.pe(x)

        for i, layer in enumerate(self.encoder):
            x = layer(x)

        out = x
        x = x[-1]

        similar_label = similars[:,:,-2]
        
        mode_values = torch.mode(similar_label, dim=1).values # torch.Size([908])
        similar_label = self.embedding(mode_values.int()) # torch.Size([890, 16])

        output = self.moe(inp = out.permute(1, 0, 2), gate_inp = similar_label)        
        return output[:,-1]

class TRA(nn.Module):

    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
    """

    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info

        if num_states > 1:
            self.router = nn.LSTM(
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        self.predictors1 = nn.Linear(input_size, hidden_size)
        self.predictors2 = nn.Linear(hidden_size, num_states)
        self.relu=nn.ReLU()

    def forward(self, hidden, hist_loss=None):
        preds = self.predictors2(self.relu(self.predictors1(hidden)))
        return preds.squeeze(-1), preds, None


def evaluate(pred):
    # pred = pred.rank(pct=True)  # transform into percentiles
    pred = pred.dropna(subset=['label'])
    score = pred.score
    label = pred.label
    diff = score - label
    MSE = (diff**2).mean()
    MAE = (diff.abs()).mean()
    IC = score.corr(label)
    # return {"MSE": MSE.astype(np.float64), "MAE": MAE.astype(np.float64), "IC": IC.astype(np.float64)}
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError("the %d-th model has different params" % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q
