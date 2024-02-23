import torch
import torch.nn as nn


# Regression Model
class Lambda_Polarization_Polynomial_Regression_BASE(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.poly = nn.Linear(degree*2, 1)  # Since we have two input features (p and lambda)
        self.degree = degree

    def forward(self, x):
        # x is expected to be of shape [batch_size, 2], with columns [p, lambda]
        poly_features = torch.cat([x[:, i:i+1]**j for i in range(2) for j in range(1, self.degree+1)], dim=1)
        # print(poly_features.shape)
        return self.poly(poly_features)