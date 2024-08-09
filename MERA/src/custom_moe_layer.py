r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from fmoe.layers import FMoE, _fmoe_general_global_forward
from fmoe.linear import FMoELinear
from functools import partial
import tree
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NaiveGate

from src.noisy_gate import NoisyGate
from src.noisy_gate_vmoe import NoisyGate_VMoE

from pdb import set_trace
import numpy as np


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # torch.Size([1, 64, 20])

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]  # torch.Size([1280, 384]) 类似gather
    feature = feature.view(batch_size, num_points, k, num_dims)  # torch.Size([1, 64, 20, 384])
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # torch.Size([1, 768, 64, 20])

    return feature


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # 计算查询、键和值
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.hidden_dim).float())
        attention_weights = torch.softmax(scores, dim=-1)
        # 计算加权和
        weighted_values = torch.matmul(attention_weights, value)
        output = torch.sum(weighted_values, dim=1)  # 对第二维进行求和，得到32*hidden_dim的表示

        return output, attention_weights



class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_gate=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        gate=NaiveGate,
        world_size=1,
        top_k=2,
        vmoe_noisy_std=1,
        gate_return_decoupled_activation=False,
        gate_task_specific_dim=-1,
        multi_gate=False,
        regu_experts_fromtask = False,
        num_experts_pertask = -1,
        num_tasks = -1,
        regu_sem = False,
        sem_force = False,
        regu_subimage = False,
        expert_prune = False,
        prune_threshold = 0.1,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.our_d_gate = d_gate
        self.our_d_model = d_model

        self.num_expert = num_expert
        self.regu_experts_fromtask = regu_experts_fromtask
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        self.regu_subimage = regu_subimage
        self.expert_prune = expert_prune
        self.prune_threshold = prune_threshold
        if self.sem_force:
            self.force_id=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]
        if self.regu_experts_fromtask:
            self.start_experts_id=[]
            start_id = 0
            for i in range(self.num_tasks):
                start_id = start_id + int(i* (self.num_expert-self.num_experts_pertask)/(self.num_tasks-1))
                self.start_experts_id.append(start_id)
            print('self.start_experts_id',self.start_experts_id)

        # self.experts = _Expert(
        #     num_expert, d_model, d_hidden, activation, rank=expert_rank
        # )
        self.e=nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
        ) 
        self.experts = nn.ModuleList([self.e for i in range(num_expert)])
        # self.experts = nn.ModuleList([nn.GRU(
        #     input_size=d_model,
        #     hidden_size=d_model,
        #     num_layers=2,
        #     batch_first=True,
        # )  for i in range(num_expert)])

        self.gate_task_specific_dim = gate_task_specific_dim
        self.multi_gate = multi_gate
        
        print('multi_gate',self.multi_gate)
        if gate == NoisyGate:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=self.regu_sem,sem_force = self.sem_force)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks)
        elif gate == NoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation,
                    noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation,
                noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)

        else:
            raise ValueError("No such gating type")
        self.mark_parallel_comm(expert_dp_comm)

        self.count = [0]*num_expert
        self.score = torch.zeros(1, num_expert).cuda()


    def my_expert_fn(self, inp, fwd_expert_count):

        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            if batch_size == 0:
                continue
            inp_slice = inp[base_idx : base_idx + batch_size]
            out,_=self.experts[i](inp_slice)
            # out=self.experts[i](inp_slice)
            outputs.append(out)
            base_idx += batch_size
        return torch.cat(outputs, dim=0)
    

    def forward(self, inp: torch.Tensor, src_mask=None, is_causal=None, src_key_padding_mask=None, gate_inp=None, task_id = None, task_specific_feature = None, sem=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        if (task_id is not None) and (task_specific_feature is not None):
            assert self.multi_gate is False
            size = gate_inp.shape[0]
            gate_inp = torch.cat((gate_inp,task_specific_feature.repeat(size,1)),dim=-1)
        output = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, sem=sem)
        return output


    def forward_moe(self, gate_inp, moe_inp, task_id=None, sem=None):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)
            tree.map_structure(ensure_comm_func, gate_inp)
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)

        if (task_id is not None) and self.multi_gate:
            # print('in custom moe_layer,task_id',task_id)
            gate_top_k_idx, gate_score = self.gate[task_id](gate_inp)
        else:
            gate_top_k_idx, gate_score = self.gate(gate_inp)
        # print(gate_top_k_idx)
        # print(gate_score)
        # self.score += torch.sum(gate_score, dim=0, keepdim=True)

        # 统计expert的使用频率
        # from collections import Counter
        # counts = Counter(gate_top_k_idx.reshape(-1,1))
        # for num, count in counts.items():
        #     self.count[num]+=count

        if self.expert_prune:
            gate_score = torch.where(gate_score>self.prune_threshold,gate_score,0.)
            prune_prob = 1-torch.nonzero(gate_score).shape[0]/torch.cumprod(torch.tensor(gate_score.shape),dim=0)[-1]
            print('prune_prob',prune_prob)
        
        if self.sem_force and (sem is not None):
            batch = sem.shape[0]
            gate_top_k_idx = gate_top_k_idx.reshape(batch,-1,self.top_k)
            sem = sem.reshape(batch,-1)
            for k in range(batch):
                for i in range(sem.shape[-1]):
                    for j in range(len(self.force_id)):
                        if sem[k,i] in self.force_id[j]:
                            gate_top_k_idx[k,i+1,:]=[j*2,j*2+1]
            gate_top_k_idx = gate_top_k_idx.reshape(-1,self.top_k)
            gate_score =  torch.ones((gate_score.shape[0],self.top_k),device=gate_score.device)*0.5

        if self.regu_experts_fromtask and (task_id is not None):
            # print('task_id',self.start_experts_id[task_id],task_id)
            gate_top_k_idx = gate_top_k_idx + self.start_experts_id[task_id]

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size)
        fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.my_expert_fn, self.num_expert, self.world_size)

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            def view_func(tensor):
                tensor = tensor.view(moe_inp_batch_size[0], self.top_k, tensor.shape[1], tensor.shape[2])
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, self.top_k, 1, 1)

        moe_outp = torch.sum(moe_outp*gate_score, dim=1) 

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp
