import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_data


def plot_distribution(y):
    dist = np.zeros(4)

    for t in range(4):
        dist[t] = (y == t).sum()

    plt.title("Dataset Distribution")
    plt.bar(["Running", "Walking", "Cycling", "Empty"], dist)
    plt.show()


if __name__ == "__main__":
    X, y = load_data('./dataset_0512_5_linke')

    # plot_distribution(y)

    type_args = []

    for t in range(4):
        type_args.append(np.argwhere(y == t).squeeze())

    generated_X = np.zeros((0, 3, 125))
    generated_y = np.zeros((0,))

    for idx in type_args:
        total = idx.size

        copy_x = np.zeros((0, 3, 125))
        copy_y = np.zeros((0,))

        while total <= 5000:
            generated_X = np.concatenate((generated_X, X[idx] + np.random.normal(-0.1, 0.1)), 0)
            generated_y = np.concatenate((generated_y, y[idx]), 0)
            total += idx.size

    X = np.concatenate((X, generated_X), 0)
    y = np.concatenate((y, generated_y), 0)

    plot_distribution(y)
