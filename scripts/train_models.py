import numpy as np
import torch
import torch.nn as nn
from scripts.GAT import GAT
from torch_geometric.loader import DataLoader
from scripts.utils import train_model, evaluate, get_data, compute_edge_index


def train_gat(batch_size=32):
    # Input Parameters
    in_channels, hidden_channels = 3, 128
    num_layers, num_heads = 5, 3
    output_dim = 8  # 7 (if the dataset is `fer2013`

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=output_dim, num_layers=num_layers, heads=num_heads)

    train_data, val_data, test_data = get_data()

    # Train Loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    edge_index = compute_edge_index()

    # Label Weight
    label_counts = np.bincount([data.y.item() for data in train_data])
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    epochs = 500

    # Trains the model
    train_losses, val_losses, accuracies = train_model(model, train_loader, val_loader, optimizer, criterion, epochs)

