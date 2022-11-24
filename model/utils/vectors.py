import torch
from torch import Tensor
from typing import Tuple

def vectorize(mgrid:Tuple[Tensor, Tensor]):
    """
    Turn meshgrid into tensor of [x,y]
    """
    x = torch.reshape(mgrid[0], (mgrid[0].numel(), 1))
    y = torch.reshape(mgrid[1], (mgrid[1].numel(), 1))
    return torch.stack((x,y))
