import numpy as np
from torch import Tensor
import torch
from typing import List, Tuple, Union
from os import path as osp

from utils import vectors


class World:
    # CLASS ATTRIBUTES
    POSSIBLE_MAP_TYPES = ["Uniform", "Gaussian", "Random"]

    def __init__(self,
                seed:int=None, 
                max_nutrient_density:float=20, 
                size:Union[List[int], Tuple[int]]=(20, 20),
                res:float=1.,
                nutrient_map_type:str="Uniform",
                device:str="cpu"):

        # Input validation
        assert(nutrient_map_type in World.POSSIBLE_MAP_TYPES or osp.isfile(nutrient_map_type))
        assert size[0]%res==0.0 and size[1]%res==0.0, "World size must be a whole-number multiple of res"

        self.seed = seed
        self.max_nutrient_density = max_nutrient_density
        self.size = tuple(size)
        self.res = res
        self.nutrient_map_type = nutrient_map_type
        self.device = device

    def make_world_coordinates(self):
        world_tensor_size = (int(self.size[0]//self.res), int(self.size[1]//self.res))
        world_tensor_numel = world_tensor_size[0] * world_tensor_size[1]
        
        self.coordinates =\
                torch.stack(torch.meshgrid(self.size[0]/2*torch.linspace(-1,1,world_tensor_size[0]),
                 self.size[1]/2*torch.linspace(-1,1,world_tensor_size[1])))
        self.coordinates_vector = vectors.vectorize((self.coordinates[0,:,:], self.coordinates[1,:,:]))

        pdist = torch.nn.PairwiseDistance(p=2)
        self.pairwise_distances = pdist(self.coordinates_vector, self.coordinates_vector)

    def generate_field(self):
        if self.nutrient_map_type == "Uniform":
            self.nutrient_map = self.max_nutrient_density * self.res**2 *\
             torch.ones(world_tensor_size, device=self.device)
            
        else:
            assert(0==1)
            

