import torch
import numpy as np
import torch.nn as nn


# Trains the model and returns the average loss
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        x, label = data.x.to(torch.float), data.y.to(torch.float)
        edge_index = data.edge_index.to(torch.float)

        x_hat = model(x, edge_index)
        loss = criterion(x_hat, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(loader)


# Evaluates the model and returns the average loss
# and accuracy
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, accuracy = [], 0

    with torch.no_grad():
        for data in loader:
            x, label = data.x.to(torch.float), data.y.to(torch.float)
            edge_index = data.edge_index.to(torch.float)
            x_hat = model(x, edge_index)
            loss = criterion(x_hat, label)
            total_loss += loss.item()

            if np.argmax(x_hat) == np.argmax(label):
                accuracy += 1
    
    return total_loss / len(loader), accuracy / len(loader)


# Returns three lists: `train_losses`, `val_losses`, `accuracies`
# The Losses contain lists of training and validation losses of the
# model over the course of epochs
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, accuracy = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
    
    return train_losses, val_losses, accuracies