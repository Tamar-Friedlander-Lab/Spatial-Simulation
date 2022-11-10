from model.world import World
import torch
from torch import Tensor

class Distribution(torch.distributions.nor):
    def __init__(self, type:str="Gaussian"):
        self.type = type
