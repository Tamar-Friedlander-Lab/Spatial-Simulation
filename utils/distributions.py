from torch import Tensor
import torch
from world import World

class Distribution:
    def __init__(self, world:World):
        self.x_limits = 0