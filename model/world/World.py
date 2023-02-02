import numpy as np
from torch import Tensor
import torch
from typing import List, Tuple, Union
from os import path as osp

from model.utils import vectors


class World:
    # CLASS ATTRIBUTES
    POSSIBLE_MAP_TYPES = ["Uniform", "Gaussian", "Random", "Rock"]

    def __init__(self,
                 seed: int = None,
                 max_nutrient_density: float = 20,
                 size: int = 21,
                 res: float = 1.,
                 nutrient_map_type: str = "Uniform",
                 device: int = None):

        # Input validation
        self.nutrient_map = None
        self.pairwise_distances = None
        self.coordinates_vector = None
        self.coordinates = None
        assert (nutrient_map_type in World.POSSIBLE_MAP_TYPES or osp.isfile(nutrient_map_type))
        assert size % res == 0.0, "World size must be a whole-number multiple of res"

        self.seed = seed
        self.max_nutrient_density = max_nutrient_density
        self.size = (size, size)
        self.res = res
        self.nutrient_map_type = nutrient_map_type
        if device is None:
            device_name = "cpu"
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.set_device(device)
            device_name = f"cuda:{device}"
        self.device_name = device_name
        self.device = torch.device(device_name)

    def make_world_coordinates(self, world_tensor_size):
        world_tensor_numel = world_tensor_size[0] * world_tensor_size[1]

        self.coordinates = \
            torch.stack(torch.meshgrid(self.size[0] / 2 * torch.linspace(-1, 1, world_tensor_size[0]),
                                       self.size[1] / 2 * torch.linspace(-1, 1, world_tensor_size[1])))
        self.coordinates_vector = vectors.vectorize((self.coordinates[0, :, :], self.coordinates[1, :, :]))

        pdist = torch.nn.PairwiseDistance(p=2)
        self.pairwise_distances = pdist(self.coordinates_vector, self.coordinates_vector)

    def generate_field(self):
        world_tensor_size = (int(self.size[0] // self.res), int(self.size[1] // self.res))
        self.make_world_coordinates(world_tensor_size)
        if self.nutrient_map_type == "Uniform":
            self.nutrient_map = self.max_nutrient_density * self.res ** 2 * \
                                torch.ones(world_tensor_size, device=self.device)
        elif self.nutrient_map_type == "Rock":
            self.nutrient_map = self.max_nutrient_density * self.res ** 2 * \
                                torch.ones(world_tensor_size, device=self.device)
            self.nutrient_map = self.nutrient_map.where(torch.sqrt(torch.sum(self.coordinates ** 2, 0)) >= 3,
                                                        torch.tensor(0.0))
        else:
            assert (0 == 1)
