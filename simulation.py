from model.world import Population, World
import time
from matplotlib import pyplot as plt
import torch

fig, ax = plt.subplots()

w = World(nutrient_map_type="Uniform", size=(101, 101))
p = Population(world=w, init_population="Gaussian", num_types=3)
p.make_population()
p.show_population(ax=ax)
for i in range(100):
    p.advance()
    p.show_population(ax=ax)
