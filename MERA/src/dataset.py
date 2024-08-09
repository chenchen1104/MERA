import numpy as np
import torch
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def collate_fn(batch):
    date = []
    feature = []
    similar = []
    yraw = []
    ynorm = []

    for sample in batch:
        date += sample["date"]
        feature.extend(sample["feature"])
        similar.extend(sample["similar"])
        yraw.extend(sample["yraw"])
        ynorm.extend(sample["ynorm"])

    return torch.FloatTensor(np.stack(feature, axis=0)), torch.FloatTensor(np.stack(similar, axis=0)), torch.FloatTensor(np.stack(yraw, axis=0)), torch.FloatTensor(np.stack(ynorm, axis=0)), date


class MinDataset(Dataset):
    def __init__(self, dates, yraw, ynorm, feature, similar):
        self.dates = dates
        self.yraw = yraw
        self.ynorm = ynorm
        self.feature = feature
        self.similar = similar
        
    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        feature = torch.FloatTensor(self.feature[date])
        similar = torch.FloatTensor(self.similar[date])
        yraw = torch.FloatTensor(self.yraw[date])
        ynorm = torch.FloatTensor(self.ynorm[date])
        
        return {
            "date": [date for _ in range(len(yraw))],
            "feature": feature,
            "similar": similar,
            "yraw": yraw,
            "ynorm": ynorm,
        }
