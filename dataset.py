import os
import random
import numpy as np

import torch
import torch.utils.data

_synsetid_to_cate = {
    "02691156": "airplane",
    "02958343": "car",
    "03001627": "chair",
}
_cate_to_synsetid = {v: k for k, v in _synsetid_to_cate.items()}


class ShapeNet15k(torch.utils.data.Dataset):
    def __init__(self, root, cate, split, random_sample, sample_size):
        self.data = []
        cate_dir = os.path.join(root, _cate_to_synsetid[cate], split)
        for fname in os.listdir(cate_dir):
            if fname.endswith(".npy"):
                path = os.path.join(cate_dir, fname)
                sample = np.load(path)[np.newaxis, ...]
                self.data.append(sample)

        self.data = np.concatenate(self.data)
        self.mu = self.data.reshape(-1, 3).mean(axis=0).reshape((1, 3))
        self.std = self.data.reshape(-1).std(axis=0).reshape((1, 1))
        self.data = (self.data - self.mu) / self.std

        self.random_sample = random_sample
        self.sample_size = sample_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        indices = (
            np.random.choice(x.shape[0], self.sample_size)
            if self.random_sample
            else np.arange(self.sample_size)
        )
        x = torch.from_numpy(x[indices]).float()
        mu = torch.from_numpy(self.mu).float()
        std = torch.from_numpy(self.std).float()
        return x, mu, std
