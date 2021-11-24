from pathlib import Path
from re import findall

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LinkeDataset(Dataset):
    def __init__(self, samples, labels, imbacenced=False):
        self.samples = samples
        self.labels = labels
        if imbacenced:
            self.imbalanced_dataset()

    def imbalanced_dataset(self):
        X = self.samples
        y = self.labels

        type_args = []

        for t in range(4):
            type_args.append(np.argwhere(y == t).squeeze())

        generated_X = np.zeros((0, 3, 125))
        generated_y = np.zeros((0,))

        for idx in type_args:
            total = idx.size

            while total <= 5000:
                generated_X = np.concatenate((generated_X, X[idx] + np.random.normal(-0.05, 0.05)), 0)
                generated_y = np.concatenate((generated_y, y[idx]), 0)
                total += idx.size

        X = np.concatenate((X, generated_X), 0)
        y = np.concatenate((y, generated_y), 0)

        self.samples = X
        self.labels = y

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.labels


def load_data(fp, test=False):

    if test:
        folder = Path(fp, "validation_margin")
    else:
        folder = Path(fp, "train_margin")

    if Path(fp, f"{'test' if test else 'train'}.npz").exists():
        # load from npz cache
        data = np.load(Path(fp, f"{'test' if test else 'train'}.npz"))
        X = data["x"]
        y = data["y"]
        return X, y

    X = []
    y = []

    files = list(folder.iterdir())

    for file in tqdm(files, desc=f"loading {'test' if test else 'train'} data", unit="file"):
        x = np.loadtxt(file)
        match = findall(r"\d+", file.name)
        X.append(x.T)
        y.append(int(match[0]) - 1)

    X = np.array(X)
    y = np.array(y, dtype=np.int32)

    # save as npz format
    np.savez(Path(fp, f"{'test' if test else 'train'}.npz"), x=X, y=y)

    return X, y


def create_data_loader(fp, batch_size=32, test=False):
    X, y = load_data(fp, test)
    dataset = LinkeDataset(X, y, not test)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(dataset)
    )
    return data_loader
