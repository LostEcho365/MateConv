import numpy as np
import torch

weight = torch.randn(2,3,2,2)
print(weight.shape)

weight = torch.view_as_complex(weight)
print(weight.shape)

weight = torch.nn.functional.pad(
    weight, pad=[0] * int((weight.dim() - 1) * 2) + [0, 1], value=1
)
#
print(weight.shape)
print(weight)
# print(weight.shape)
#
print(torch.arange(weight.shape[1]))
print(weight[[0, 1, 2]])
weight = weight[torch.arange(weight.shape[1])]
print(weight.shape)
print(weight)


# x = torch.tensor([[1, 2], [3, 4], [5, 6]])
# rows = torch.tensor([0, 2])
# cols = torch.tensor([0, 1])
# y = x[2, 1]
# print(y)  # This will be a tensor with values [1, 6]


#
# pad=[0] * int((weight.dim() - 1) * 2) + [0, 1]
# print(pad)

# # Real-valued tensor with shape [2, 2, 2]
# # Each [2] at the end is [real_part, imaginary_part]
# real_tensor = torch.tensor([
#     [[1., 2.], [3., 4.]],
#     [[5., 6.], [7., 8.]]
# ])
#
#
# # Convert to complex tensor
# complex_tensor = torch.view_as_complex(real_tensor)

# print(complex_tensor)
