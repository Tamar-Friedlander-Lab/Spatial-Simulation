# Imports
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from torch import Tensor
import time

from model.utils import Distribution

from .World import World


# matplotlib.use("tkagg")
# plt.interactive(False)


class Population:
    """
    TODO:
    1) Implement Gaussian, circles and general densities for population
    2) Remove hard-coded Gaussian generation from MakePopulation().
    """

    # CLASS ATTRIBUTES
    POSSIBLE_POPULATION_TYPES = ["Gaussian", "Circles", "Half", "Test", "Test2"]

    def __init__(self,
                 num_types: int = 2,
                 seed: int = None,
                 world: World = None,
                 init_population="Gaussian",
                 compatibility=None,
                 self_pollen=0.8,
                 inbreeding_depression=0.5,
                 seed_rate=1,
                 max_seeds=20):

        # Input validation
        self.total_population = None
        self.population = None
        assert num_types > 1, "num_types must be an integer greater than 1"
        assert init_population in Population.POSSIBLE_POPULATION_TYPES, "Population type error"

        self.num_types = num_types
        self.seed = seed
        if world is not None:
            self.world = world
        else:
            self.world = World()
        self.world.generate_field()

        self.init_population = init_population
        self.seed_rate = seed_rate
        self.max_seeds = max_seeds
        self.self_pollen = self_pollen
        self.inbreeding_depression = inbreeding_depression
        if compatibility is None:
            compatibility = 1 - torch.eye(self.num_types)
        self.compatibility = torch.unsqueeze(torch.unsqueeze(compatibility, -1), -1)
        self.generation = 0

        pollen_kernel = torch.exp(-torch.sqrt(torch.sum(self.world.coordinates ** 2, 0)) / 2)
        self.pollen_func = Population.make_pollen_function(pollen_kernel, self.world.device)

        seed_kernel = torch.ones(3, 3)
        seed_kernel[1, 1] = self.max_seeds - 8
        seed_kernel = seed_kernel / seed_kernel.sum()
        self.seed_func = Population.make_seed_function(seed_kernel, self.world.device)
        # self.pollen_pdf = self.pollen_kernel(self.world.pairwise_distances)

    @staticmethod
    def make_pollen_function(kernel: Tensor, device):
        pollen_func = torch.nn.Conv2d(1, 1, kernel.shape, bias=False, padding="same", device=device)
        pollen_func.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(kernel, 0), 0), requires_grad=False)
        return pollen_func

    @staticmethod
    def make_seed_function(kernel: Tensor, device):
        seed_func = torch.nn.Conv2d(1, 1, kernel.shape, bias=False, padding="same", device=device)
        seed_func.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(kernel, 0), 0), requires_grad=False)
        return seed_func

    def make_population(self):
        if self.init_population == "Gaussian":
            self.make_population_gaussians()
        elif self.init_population == "Circles":
            self.make_population_circles()
        elif self.init_population == "Half":
            self.make_population_half()
        elif self.init_population == "Test":
            # One plant of type 0 became SC on the male side
            self.num_types = 3
            tmp = 1 - torch.eye(3)
            tmp[2, 2] = 1
            self.compatibility = torch.unsqueeze(torch.unsqueeze(tmp, -1), -1)
            self.population = torch.zeros(3, self.world.size[0], self.world.size[1])
            self.population[0:2, :, :] = self.world.nutrient_map
            mid = self.world.size[0] // 1 * 0
            self.population[2, mid, mid] += 2
            self.population[0:2, mid, mid] -= 1
        elif self.init_population == "Test2":
            self.num_types = 3
            tmp = 1 - torch.eye(3)
            tmp[2, 2] = 1
            self.compatibility = torch.unsqueeze(torch.unsqueeze(tmp, -1), -1)
            distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :]) ** 2 +
                                                (self.world.coordinates[1, :, :]) ** 2)
                                     for i in range(3)])
            # Generate population
            self.population = self.world.max_nutrient_density * torch.exp(-distances ** 2 / 100)
            self.population[2, :, :] = 0
            # mid = [35, 54]
            mid = [50, 50]
            self.population[2, mid, mid] += 1
            self.population[0, mid, mid] -= 1
        self.population = self.carrying_capacity(self.population)
        self.total_population = self.population.sum(0)
        self.show_init_population(block=False)

    def make_population_gaussians(self):
        # Compute distances from each center
        if self.num_types == 2:
            centers = [Tensor([-10, 0]), Tensor([10, 0])]
        elif self.num_types == 3:
            centers = [Tensor([-25, 0]), Tensor([25, 0]), Tensor([0, 25]), Tensor([0, -25])]
        distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :] - centers[i][0]) ** 2 + \
                                            (self.world.coordinates[1, :, :] - centers[i][1]) ** 2) \
                                 for i in range(len(centers))])
        # Generate population
        self.population = 20 * torch.exp(-distances ** 2 / 100)
        if self.num_types == 3:
            self.population[2, :, :] = self.population[2, :, :] + self.population[3, :, :] * 0
            self.population = self.population[:3, :, :]

    def make_population_circles(self):
        # Compute distances from each center
        if self.num_types == 2:
            centers = [Tensor([-25, 0]), Tensor([25, 0])]
        elif self.num_types == 3:
            centers = [Tensor([-25, 0]), Tensor([25, 0]), Tensor([0, 25])]
        distances = torch.stack([torch.sqrt((self.world.coordinates[0, :, :] - centers[i][0]) ** 2 + \
                                            (self.world.coordinates[1, :, :] - centers[i][1]) ** 2) \
                                 for i in range(len(centers))])

        # Generate population
        self.population = distances.where(distances <= 10, torch.tensor([.0]))
        self.population = self.population.where(self.population == 0.0, torch.tensor(self.max_seeds))

    def make_population_half(self):
        # Generate population
        self.population = torch.stack([self.world.nutrient_map for i in range(self.num_types)])
        self.population[0, :, :] = self.population[0, :, :].where(self.world.coordinates[1, :, :] > 0,
                                                                  torch.tensor(0.0))
        self.population[1, :, :] = self.population[1, :, :].where(self.world.coordinates[1, :, :] < 0,
                                                                  torch.tensor(0.0))

    def compute_pollen(self):
        return torch.squeeze(self.pollen_func(torch.unsqueeze(self.population, 1)))

    def compute_offspring(self, interactions: Tensor, outer_pollen: Tensor, self_pollen: Tensor):
        total_pollen = outer_pollen.sum(0) + self_pollen
        pollen_weighted_interactions = (outer_pollen / total_pollen * interactions).where(total_pollen != 0,
                                                                                          torch.tensor(0.0))
        i_is_female = pollen_weighted_interactions.sum(0)
        i_is_male = pollen_weighted_interactions.sum(1)
        i_is_self = ((1 - self.inbreeding_depression) * self_pollen / total_pollen * self.population).where(
            total_pollen != 0, torch.tensor(0.0))
        # i_is_both = outer_pollen.diagonal(dim1=0, dim2=1).permute(2, 0, 1) / total_pollen * interactions.diagonal(
        #     dim1=0, dim2=1).permute(2, 0, 1)
        return (0.5 * i_is_female.where(~i_is_female.isnan() != 0, torch.tensor(0.0)) \
                + 0.5 * i_is_male.where(~i_is_male.isnan() != 0, torch.tensor(0.0)) \
                + i_is_self.where(~i_is_self.isnan() != 0, torch.tensor(0.0))) * self.seed_rate

    def disperse_seeds(self, new_seeds):
        return torch.squeeze(self.seed_func(torch.unsqueeze(new_seeds, 1)))

    def carrying_capacity(self, new_seeds: Tensor):
        regulated = new_seeds.where(self.world.nutrient_map >= torch.sum(new_seeds, 0),
                                    self.world.nutrient_map * new_seeds / torch.sum(new_seeds, 0))
        regulated = regulated.where(torch.sum(new_seeds, 0) > 0.01, torch.tensor(0.0))
        return regulated.floor()

    def advance(self):
        # interactions[bottom, top, x, y]
        pollen_dist = self.compute_pollen() * (1 - self.self_pollen)
        outer_pollen = (self.compatibility * pollen_dist.unsqueeze(1) / self.total_population).where(
            self.total_population != 0, torch.tensor(0.0))
        self_pollen = self.self_pollen * self.compatibility.diagonal(dim1=0, dim2=1).permute(2, 0, 1)

        interactions = torch.unsqueeze(pollen_dist, 1) * (self.population / self.total_population)  # I^[1]_[0] [2,3]
        # interactions[range(self.num_types), range(self.num_types), :, :] = 0 #I^i_i = 0
        interactions = interactions.where(self.total_population != 0, torch.tensor(0.0))

        new_seeds = self.compute_offspring(interactions, outer_pollen, self_pollen)
        new_seeds = self.disperse_seeds(new_seeds)
        self.population = self.carrying_capacity(new_seeds)

        self.generation += 1

        assert (torch.all(torch.sum(self.population, 0) <= self.world.nutrient_map))

    @staticmethod
    def show_tensor(tensor: Tensor, result=False):
        if tensor.shape[0] == 2:
            population_image = \
                torch.stack(
                    [*(tensor / tensor.max()), torch.zeros_like(tensor[0, :, :].squeeze())])
        else:
            population_image = tensor.squeeze()
        if len(tensor.shape) == 3:
            population_image = population_image.permute((1, 2, 0))
        print(population_image.shape)
        fig, ax = plt.subplots()
        ax.imshow(population_image)
        plt.show(block=False)
        if result:
            return population_image

    def show_population(self, out_file=None, block=False, ax: plt.Axes = None):
        if self.num_types == 2:
            population_image = \
                torch.permute(
                    torch.stack(
                        [*(self.population / self.world.nutrient_map), torch.zeros_like(self.world.nutrient_map)]),
                    (1, 2, 0))
        elif self.num_types == 3:
            population_image = torch.permute(self.population / self.world.nutrient_map, (1, 2, 0))
        if out_file is not None:
            plt.imsave(out_file, population_image)
        else:
            assert len(plt.get_fignums()) != 0
            ax.clear()
            ax.imshow(population_image)
            ax.set_title(f"Generation {self.generation}")
            if block:
                plt.show(block=block)
            else:
                plt.pause(1.5)

    def show_init_population(self, block=False):
        if self.num_types == 2:
            population_image = \
                torch.permute(
                    torch.stack(
                        [*(self.population / self.world.nutrient_map), torch.zeros_like(self.world.nutrient_map)]),
                    (1, 2, 0))
        elif self.num_types == 3:
            population_image = torch.permute(self.population / self.world.nutrient_map, (1, 2, 0))
        fig, ax = plt.subplots()
        ax.imshow(population_image)
        ax.set_title("Initial Population")
        plt.show(block=block)
