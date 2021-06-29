import json

import torch
from torch.utils.data import Dataset


class SkDataset(Dataset):
    def __init__(self, json_file: str,):
        self.data = []
        with open(json_file) as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = self.data[idx]['en']
        label = self.data[idx]['bleu']

        return text, torch.tensor(label)