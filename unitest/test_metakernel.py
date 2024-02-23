'''
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-28 18:03:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-08-28 18:20:40
'''
import torch
def test_metakernel():
    width = 4
    depth = 3
    size = 2
    channel = 2
    weight = torch.randn(width,depth,size, size)
    indices = torch.randint(0, width, size=[5, channel, depth])
    out = weight[indices,torch.arange(weight.size(1))]
    print(out)
    print(weight.shape, indices.shape)
    print(out.shape)

test_metakernel()