import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from core.models.layers.phi_linear import Lambda_Polarization_Polynomial_Regression_BASE
import numpy as np

def load_data(lut_filename='../2D_LUT.npy'):
    data = np.load(lut_filename)
    p_values, lambda_values = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(1.35, 1.55, 100))
    p_values = p_values.reshape(-1)
    # print(p_values)
    lambda_values = lambda_values.reshape(-1)
    # print(lambda_values)
    phi_values = data.reshape(-1)
    dataset = TensorDataset(torch.Tensor(np.vstack([p_values, lambda_values]).T), torch.Tensor(phi_values))
    return dataset

# Splitting Data
def split_data(dataset, train_frac=0.8, batch_size=1024):
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train(model, train_loader, optimizer, criterion, epochs=300):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}")


max_degree = 10
best_degree = 1
best_val_loss = float('inf')
best_model = None
train_loader, val_loader = split_data(load_data('../2D_LUT.npy'), batch_size=32)

# print(f"train_loader: {train_loader[0]}")


for degree in range(1, max_degree + 1):
    model = Lambda_Polarization_Polynomial_Regression_BASE(degree=degree)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    print(f"Training model with polynomial degree {degree} ...")
    train(model, train_loader, optimizer, criterion)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Check if current model is the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_degree = degree
        best_model = model
        torch.save(best_model.state_dict(), 'best_model.pth')
print(f"Best polynomial degree is: {best_degree}")