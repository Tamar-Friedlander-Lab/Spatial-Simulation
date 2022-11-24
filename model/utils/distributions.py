from model.world import World
import torch
from torch import Tensor

class Distribution(torch.Tensor):
    def __init__(self, type:str="Gaussian"):
        self.type = type
