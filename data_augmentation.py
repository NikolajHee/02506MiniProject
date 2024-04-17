'''
This script performs a selection of data augmentation types, e.g. mirroring the image. 
'''

import numpy as np
import torch 

def mirror_input(input, dim: int):
    if not isinstance(dim, int):
        raise TypeError("dimension argument must be an integer: either 0 (along the horizontal axis) or 1 (along the vertical axis)")
    
    return torch.flip(input, [dim])

def mirror_horizontal(input):
    return mirror_input(input, 0)

def mirror_vertical(input):
    return mirror_input(input, 1)






# Testing:
x = torch.arange(8).view(4, 2)


# print(x)
# print(mirror_horizontal(x))


