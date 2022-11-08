# Imports
from inspect import _void
import numpy as np
from torch import Tensor
import torch
from matplotlib import pyplot as plt
from typing import List, Tuple, Union

from .World import World

class Population:

    # TODO:
    # 1) Implement Gaussian, circles and general densities for population
    # 2) Remove hard-coded Gaussian generation from MakePopulation().

    def __init__(self, 
                num_types:int=2, 
                seed:int=None, 
                world:World=None,
                densities="Gaussian"):
        
        # Input validation
        assert(num_types>1, "num_types must be an integer greater than 1")

        self.num_types = num_types
        self.seed = seed
        if world is not None:
            self.world = world
        else:
            self.world = World()
        self.world.GenerateField()
        self.densities = densities

    def MakePopulation(self):
        if self.densities == "Gaussian":
            self.MakePopulationGaussians()

    def MakePopulationGaussians(self):
        # Compute distances from each center
        centers = [Tensor([-4, 0]), Tensor([-4, 0])]
        distances = torch.stack([torch.sqrt((self.world.coordinates[0,:,:]-centers[i][0])**2+\
            (self.world.coordinates[1,:,:]-centers[i][1])**2)\
            for i in range(len(centers))])
        
        # Generate population
        self.population = 20 * torch.exp(-(distances**2)/20)
    
    def ShowPopulation(self, out_file = None):
        if self.num_types==2:
            population_image = \
                torch.permute(\
                    torch.stack([*(self.population / self.world.nutrient_map), torch.zeros_like(self.world.nutrient_map)]),
                    (1,2,0)).numpy()
        if out_file is not None:
            plt.imsave(out_file, population_image)
            