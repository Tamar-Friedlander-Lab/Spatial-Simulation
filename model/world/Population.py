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

    # CLASS ATTRIBUTES
    POSSIBLE_POPULATION_TYPES = ["Gaussian", "Circles"]
 
    def __init__(self,
                 num_types: int = 2,
                 seed: int = None,
                 world: World = None,
                 init_population="Gaussian",
                 pollen_range = None,
                 pollen_res = None,
                 seed_rate = 1,
                 max_seeds = 20):

        # Input validation
        self.population = None
        assert(num_types > 1, "num_types must be an integer greater than 1")
        assert(init_population in Population.POSSIBLE_POPULATION_TYPES, "Population type error")

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

        self.init_population = init_population
        self.seed_rate = seed_rate
        self.max_seeds = max_seeds

        self.generation = 0

        pollen_kernel = torch.exp(-torch.sqrt(torch.sum(self.world.coordinates**2, 0))/2)
        self.pollen_func = self.make_pollen_function(pollen_kernel, self.world.device)

        seed_kernel = torch.ones(3,3)
        seed_kernel[1,1] = self.max_seeds - 8
        self.seed_func = self.make_seed_function(seed_kernel, self.world.device)
        # self.pollen_pdf = self.pollen_kernel(self.world.pairwise_distances)

    def make_pollen_function(self, kernel:Tensor, device):
        pollen_func = torch.nn.Conv2d(1, 1, kernel.shape, bias = False, padding="same", device=device)
        pollen_func.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(kernel, 0), 0), requires_grad=False)
        return pollen_func

    def make_seed_function(self, kernel:Tensor, device):
        seed_func = torch.nn.Conv2d(1, 1, kernel.shape, bias = False, padding="same", device=device)
        seed_func.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(kernel, 0), 0), requires_grad=False)
        return seed_func

    def make_population(self):
        if self.init_population == "Gaussian":
            self.make_population_gaussians()
        elif self.init_population == "Circles":
            self.make_population_circles()
        self.population = self.carrying_capacity(self.population)

    def make_population_gaussians(self):
        # Compute distances from each center
        if self.num_types == 2:
            enters = [Tensor([-25, 0]), Tensor([25, 0])]
        elif self.num_types == 3:
            centers = [Tensor([-25, 0]), Tensor([25, 0]), Tensor([0, 25])]
        distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :] - centers[i][0]) ** 2 + \
                                            (self.world.coordinates[1, :, :] - centers[i][1]) ** 2) \
                                 for i in range(len(centers))])

        # Generate population
        self.population = 20 * torch.exp(-(distances ** 2) / 50)
    
    def make_population_circles(self):
        # Compute distances from each center
        if self.num_types == 2:
            enters = [Tensor([-25, 0]), Tensor([25, 0])]
        elif self.num_types == 3:
            centers = [Tensor([-25, 0]), Tensor([25, 0]), Tensor([0, 25])]
        distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :] - centers[i][0]) ** 2 + \
                                            (self.world.coordinates[1, :, :] - centers[i][1]) ** 2) \
                                 for i in range(len(centers))])

        # Generate population
        self.population = distances.where(distances<=10, torch.tensor([.0]))
        self.population = self.population.where(self.population==0.0, torch.tensor(self.max_seeds))
    
    # This function should not exist
    def compute_pollen(self):
        return torch.squeeze(self.pollen_func(torch.unsqueeze(self.population, 1)))

    def compute_offspring(self, interactions):
        all_female_interactions = torch.sum(interactions, 0) #I^i
        norm_male_interactions = torch.sum(interactions, 1) #I_i
        # norm_male_interactions = torch.sum(interactions / torch.unsqueeze(all_female_interactions, 1), 0)

        norm_interactions = (all_female_interactions + norm_male_interactions) / 2
        return torch.min(self.seed_rate * norm_interactions, torch.tensor(self.max_seeds))

    def disperse_seeds(self, new_seeds):
        return torch.squeeze(self.pollen_func(torch.unsqueeze(new_seeds, 1)))

    def carrying_capacity(self, new_seeds:Tensor):
        regulated = new_seeds.where(self.world.nutrient_map>torch.sum(new_seeds, 0), self.world.nutrient_map * new_seeds/torch.sum(new_seeds, 0))
        regulated = regulated.where(torch.sum(new_seeds, 0)>0.01, torch.tensor(0.0))
        return regulated

    def advance(self):
        # interactions[bottom, top, x, y]
        pollen_dist = self.compute_pollen()
        interactions = torch.unsqueeze(self.population, 1) * (pollen_dist / (torch.sum(pollen_dist, 0) - pollen_dist)) # I^[1]_[0] [2,3]
        interactions[range(self.num_types), range(self.num_types), :, :] = 0 #I^i_i = 0
        interactions = interactions.where(~torch.isnan(interactions), torch.tensor(0.0))

        new_seeds = self.compute_offspring(interactions)
        # new_seeds = self.disperse_seeds(new_seeds)
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
        elif self.num_types == 3:
            population_image = torch.permute(self.population/self.world.nutrient_map, (1, 2, 0)).numpy()
        population_image[np.isnan(population_image)] = 0.0
        assert(np.all(population_image<=1.0))
            # population_image /= np.max(population_image)
        if out_file is not None:
            plt.imsave(out_file, population_image)
        else:
            assert len(plt.get_fignums())!=0
            ax.clear()
            ax.imshow(population_image)
            ax.set_title(f"Generation {self.generation}")
            plt.pause(1)

