import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from  models.deQconv import DeformConv2d
from models.quaternion_layers import  QuaternionConv
from einops import rearrange
import models.quaternion_ops as core_qnn
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    upscale = 2
    window_size = 7
    height = 200
    width = 200

    #model = Restormer()
    # print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 8, height, width))
    # print(x.shape)
    y = QuaternionConv(8, 16, kernel_size=3,stride=1,padding=1, bias=True)(x)
    print(count_parameters(QuaternionConv(4, 4, kernel_size=3,stride=1,padding=1, bias=False)))
    print(count_parameters(nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)))
    # print(model.flops())
    #print(y.shape)
    convfunc = F.conv2d