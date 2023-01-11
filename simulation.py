from model.world import Population, World
import time
from matplotlib import pyplot as plt
import torch

fig, ax = plt.subplots()

w = World(nutrient_map_type="Uniform", size=(101, 101))
# comp_table = 1-torch.eye(2)
# p = Population(world=w, init_population="Gaussian", num_types=2, compatibility=comp_table, inbreeding_depression=0.0)
comp_table = torch.zeros(3,3)
comp_table[2,2] = 1
comp_table[:2,:2] = 1-torch.eye(2)
p = Population(world=w, init_population="Gaussian", num_types=3, compatibility=comp_table, inbreeding_depression=1.0)
p.make_population()
p.show_population(ax=ax)
for i in range(100):
    p.advance()
    p.show_population(ax=ax)
