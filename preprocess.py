from pathlib import Path
from re import findall

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LinkeDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def load_data(fp, test=False):

    if test:
        folder = Path(fp, "validation_margin")
    else:
        folder = Path(fp, "train_margin")

    X = []
    Y = []

    files = list(folder.iterdir())

    for file in tqdm(files, desc=f"loading {'test' if test else 'train'} data", unit="file"):
        x = np.loadtxt(file)
        match = findall(r"\d+", file.name)
        y = int(match[0]) - 1
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y, dtype=np.int32)

    return X, Y


def create_data_loader(fp, batch_size=64, test=False):
    X, Y = load_data(fp, test)
    dataset = LinkeDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader
