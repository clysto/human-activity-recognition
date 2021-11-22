import torch
import torch.nn as nn
import torch.optim as optim

from network import Network
from preprocess import create_data_loader


def valid(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            sample = sample.view(-1, 3, 1, 125).float()
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def train(model, optimizer, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    nepoch = 16

    for e in range(nepoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, target in train_loader:
            sample = sample.view(-1, 3, 1, 125).float()
            target = target.long()
            output = model(sample)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)
        acc_test = valid(model, test_loader)
        print(
            f"Epoch: [{e+1}/{nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {acc_test:.2f}"
        )


if __name__ == "__main__":
    model = Network()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    train_loader = create_data_loader("./dataset_0512_5_linke")
    test_loader = create_data_loader("./dataset_0512_5_linke", test=True)
    train(model, optimizer, train_loader, test_loader)
