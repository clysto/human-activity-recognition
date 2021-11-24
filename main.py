import torch
import torch.nn as nn
import torch.optim as optim

from network import Network
from preprocess import create_data_loader


def train_loop(model, optimizer, train_loader, loss_fn):
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader, start=1):
        X = X.view(-1, 3, 1, 125).float()
        y = y.long()
        output = model(X)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model, test_loader, loss_fn):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(-1, 3, 1, 125).float()
            y = y.long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(model, optimizer, train_loader, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    for t in range(32):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, optimizer, train_loader, loss_fn)
        test_loop(model, test_loader, loss_fn)
    print("Done!")


if __name__ == "__main__":
    model = Network()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    train_loader = create_data_loader("./dataset_0512_5_linke")
    test_loader = create_data_loader("./dataset_0512_5_linke", test=True)
    train(model, optimizer, train_loader, test_loader)
