"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-28 20:43:20
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-08-28 20:58:34
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.layers.meta_conv2d import MetaConv2d
from pyutils.general import TorchTracemalloc
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# def test_conv2d():
#     #device = "cuda:0"
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     # with TorchTracemalloc(verbose=False):
#     #     conv = MetaConv2d(
#     #         32, 32, kernel_size=(32, 32), path_multiplier=4, path_depth=4, w_bit=32, in_bit=32, device=device, with_cp=True,
#     #     )
#     #     conv.reset_parameters()
#     #     x = torch.randn(16, 32, 16, 16, device=device)
#     #     with torch.cuda.amp.autocast(True):
#     #         y = conv(x)
#
#     conv = MetaConv2d(
#         32, 32, kernel_size=(32, 32), path_multiplier=6, path_depth=4, w_bit=32, in_bit=32, device=device, with_cp=True,
#     )
#     conv.reset_parameters()
#     x = torch.ones(16, 1, 4, 4, device=device)
#     y = conv(x)
#     # y.get_alm_perm_loss()
#     # conv.set_gumbel_temperature(5.0)
#     # conv.path_generator(True)
#     # # print(y)
#     y.sum().backward()

    # conv.get_alm_perm_loss()


class CombinedModel(nn.Module):
    def __init__(self, device, kernel_size=3, padding=1):
        super(CombinedModel, self).__init__()
        self.metaconv_layer = MetaConv2d(
        32, 32, kernel_size=(32, 32), path_multiplier=6, path_depth=4, w_bit=32, in_bit=32, device=device, with_cp=True,
    )
        # The last conv layer after your custom layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)  # Example parameters

        # An adaptive avg pool layer to reduce the spatial dimensions to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Dense layer (fully connected) to output 10 classes for MNIST
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.metaconv_layer(x)  # Pass input through custom layer
        # print(x.shape)
        x = F.relu(self.conv2(F.relu(x)))  # Activation after the last conv layer
        x = self.adaptive_pool(x)  # Reduce spatial dimensions for the dense layer
        x = torch.flatten(x, 1)  # Flatten the output for the dense layer
        x = self.fc(x)  # Final dense layer
        return x



# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluation function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = CombinedModel(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 20):  # 5 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = CombinedModel(device)
#     model.to(device)
#     x = torch.ones(16, 1, 4, 4, device=device)
#     y = model(x)
#     print(model)
    # torch.autograd.set_detect_anomaly(True)
    # test_conv2d()
