# Imports
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor

from model.utils import Distribution

from .World import World


class Population:
    """
    TODO:
    1) Implement Gaussian, circles and general densities for population
    2) Remove hard-coded Gaussian generation from MakePopulation().
    """
 
    def __init__(self,
                 num_types: int = 2,
                 seed: int = None,
                 world: World = None,
                 densities="Gaussian",
                 pollen_range = None,
                 pollen_res = None,
                 seed_rate = 0.01,
                 max_seeds = 5):

        # Input validation
        self.population = None
        assert(num_types > 1, "num_types must be an integer greater than 1")

        self.num_types = num_types
        self.seed = seed
        if world is not None:
            self.world = world
        else:
            self.world = World()
        self.world.generate_field()
        if pollen_res is not None:
            assert(pollen_res%world.res==0.0)
        else:
            pollen_res = self.world.res

        self.densities = densities
        self.seed_rate = seed_rate
        self.max_seeds = max_seeds

        self.generation = 0

        pollen_kernel = torch.exp(-torch.sum(self.world.coordinates**2, 0)/1.5) / 6.81978
        self.pollen_func = self.make_pollen_function(pollen_kernel, self.world.device)

        # self.pollen_pdf = self.pollen_kernel(self.world.pairwise_distances)

    def make_pollen_function(self, kernel:Tensor, device):
        pollen_func = torch.nn.Conv2d(1, 1, kernel.shape, bias = False, padding="same", device=device)
        print(pollen_func.bias)
        pollen_func.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(kernel, 0), 0), requires_grad=False)
        return pollen_func

    def make_population(self):
        if self.densities == "Gaussian":
            self.make_population_gaussians()
        
        self.population = self.carrying_capacity(self.population)



    def make_population_gaussians(self):
        # Compute distances from each center
        world_size_tensor = torch.tensor(self.world.size)
        centers = (torch.rand(2,2)*2 - 1) * world_size_tensor
        distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :] - centers[i,0]) ** 2 + \
                                            (self.world.coordinates[1, :, :] - centers[i,1]) ** 2) \
                                 for i in range(len(centers))])

        # Generate population
        self.population = 10 * torch.exp(-(distances ** 2) / 10)
    
    # This function should not exist
    def compute_pollen(self):
        return torch.squeeze(self.pollen_func(torch.unsqueeze(self.population, 1)))

    def compute_offspring(self, interactions):
        all_female_interactions = torch.sum(interactions, 0) #I^i
        norm_male_interactions = torch.sum(interactions / torch.unsqueeze(all_female_interactions, 1), 0)

        norm_interactions = (all_female_interactions + norm_male_interactions) / 2
        return torch.min(self.seed_rate * norm_interactions, torch.tensor(self.max_seeds))

    def carrying_capacity(self, new_seeds):
        regulated = new_seeds * torch.min(self.world.nutrient_map/torch.sum(new_seeds, 0), torch.tensor(1.0))
        regulated = regulated.where(torch.sum(new_seeds, 0)>1e-6, torch.tensor(0.0))
        return regulated

    
    def advance(self):
        # interactions[bottom, top, x, y]
        pollen_dist = self.compute_pollen()
        interactions = torch.unsqueeze(self.population, 1) * (pollen_dist / (torch.sum(pollen_dist, 0) - pollen_dist)) # I^[1]_[0] [2,3]
        interactions[range(self.num_types), range(self.num_types), :, :] = 0 #I^i_i = 0

        new_seeds = self.compute_offspring(interactions)
        self.population = self.carrying_capacity(new_seeds)

        self.generation += 1

        assert(torch.all(torch.sum(self.population, 0) <= self.world.nutrient_map))

    def show_population(self, out_file=None, block=True, ax:plt.Axes=None):
        if self.num_types == 2:
            population_image = \
                torch.permute( \
                    torch.stack(
                        [*(self.population / self.world.nutrient_map), torch.zeros_like(self.world.nutrient_map)]),
                    (1, 2, 0)).numpy()
            population_image /= np.max(population_image)
        if out_file is not None:
            plt.imsave(out_file, population_image)
        else:
            ax.clear()
            ax.imshow(population_image)
            ax.set_title(f"Generation {self.generation}")
            plt.pause(.33)

