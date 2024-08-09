import argparse

import qlib
from ruamel.yaml import YAML
from qlib.utils import init_instance_by_config
from src.dataset import MinDataset,collate_fn
import h5py
import numpy as np
import torch

def main(seed, config_file="./configs/config_transformer.yaml"):
    # set random seed
    with open(config_file) as f:
        yaml = YAML(typ='safe', pure=True)
        config=yaml.load(f)

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        region=config["qlib_init"]["region"],
    )
    # dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    f_feature = h5py.File('./500_processed/500_features.h5')
    f_label_raw = h5py.File('./500_processed/500_label_raw.h5')
    f_label_norm = h5py.File('./500_processed/500_label_norm.h5')
    f_similar = h5py.File('./500_processed/similars50.h5') 

    features = {}
    similars = {}
    yraws = {}
    ynorms = {}
    dates = []
    for key in f_feature.keys():
        dates.append(key)
        feature = np.array(f_feature[key])
        features[key] = feature

        similar = np.array(f_similar[key])
        similars[key] = similar

        yraw = np.array(f_label_raw[key])
        yraws[key] = yraw

        ynorm = np.array(f_label_norm[key])
        ynorms[key] = ynorm

    train_dataset = MinDataset(dates=dates[:970], yraw=yraws, similar=similars, ynorm=ynorms, feature=features)
    valid_dataset = MinDataset(dates=dates[970:1116], yraw=yraws, similar=similars, ynorm=ynorms, feature=features)
    test_dataset = MinDataset(dates=dates[1116:],yraw=yraws, similar=similars, ynorm=ynorms, feature=features)

    model.fit(train_dataset, valid_dataset, test_dataset)
    # model.predict(test_dataset)


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="./configs/config_transformer.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))
