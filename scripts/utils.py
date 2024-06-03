import torch
import pickle
import scripts
import numpy as np
import networkx as nx


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
    total_loss, accuracy = 0, 0

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


def get_data():
    dataset = 'ck'
    train_data_path = dataset + '_data/train_data_70_20_10.pkl'
    val_data_path = dataset + '_data/val_data_70_20_10.pkl'
    test_data_path = dataset + '_data/test_data_70_20_10.pkl'

    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data


def compute_edge_index():
    adjacency_matrix = np.loadtxt(scripts.__path__[0] + '/../standard_mesh_adj_matrix.csv', delimiter=',')
    graph = nx.from_numpy_array(adjacency_matrix)
    edge_index = []
    for edge in graph.edges:
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()

    return edge_index
