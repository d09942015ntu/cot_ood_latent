import json
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class TrainDataset(Dataset):
    def __init__(self, data_dir, t=10, t2=1, h_position=None, ftype='train', length=1024):
        self.data = json.load(open(os.path.join(data_dir, f"{ftype}.json"), "r"))
        self.Theta_size = len(self.data.keys())
        self.t = t
        self.t2 = t2
        self.h = int(os.path.basename(data_dir).split('_')[1])
        self.rng = np.random.RandomState(0)
        self.input_length = (self.t + self.t2) * self.h
        if h_position is None:
            h_position = list(range(1, self.h))
        self.h_position = h_position
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.rng = np.random.RandomState(idx)
        theta = self.rng.choice(list(self.data.keys()))
        theta_data = self.data[theta]
        selected_idx = self.rng.choice(range(len(theta_data)), size=self.t + self.t2, replace=False)

        input_vals = []
        for idx2 in selected_idx:
            input_vals.extend(theta_data[idx2])

        input_vals = np.array(input_vals).astype(np.float32)
        label_idx = self.rng.randint(self.t - 1, self.t + self.t2 - 1) * self.h + self.rng.choice(self.h_position)
        assert label_idx % self.h != 0
        position_ids = list(range(len(input_vals)))

        labels = -100 * np.ones((self.input_length,)).astype(np.float32)
        labels[label_idx] = input_vals[label_idx]
        attention_mask = np.zeros((self.input_length,)).astype(int)
        attention_mask[:label_idx] = 1
        input_vals = input_vals[:, np.newaxis]
        labels = labels[:, np.newaxis]

        return {
            'input_ids': torch.tensor(input_vals),
            'position_ids': torch.tensor(position_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }


class TestDataset(TrainDataset):
    def __init__(self, data_dir, t=10, t2=1, h_position=None, ftype='test', length=1024):
        super().__init__(data_dir, t=t, t2=t2, h_position=h_position, ftype=ftype, length=length)


class RawDataset(TrainDataset):
    def __init__(self, data_dir, t=10, t2=1, h_position=None, ftype='test', length=1024):
        super().__init__(data_dir, t=t, t2=t2, h_position=h_position, ftype=ftype, length=length)

    def __getitem__(self, idx):
        self.rng = np.random.RandomState(idx)
        theta = self.rng.choice(list(self.data.keys()))
        theta_data = self.data[theta]
        selected_idx = self.rng.choice(range(len(theta_data)), size=self.t + self.t2, replace=False)

        input_vals = []
        for idx2 in selected_idx:
            input_vals.append(theta_data[idx2])
        return input_vals


if __name__ == '__main__':
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    data_name = 'data/discrete14_3_3_lrelu_l2_s00'
    dataset = TrainDataset(data_name)
    for item in dataset:
        for key, value in item.items():
            print(f"{key}:  {value}")
