from model.world import Population, World
import time
import matplotlib
from matplotlib import pyplot as plt
import torch

# matplotlib.use("tkagg")


w = World(nutrient_map_type="Uniform", size=101, device=None, max_nutrient_density=100)
# comp_table = 1-torch.eye(2)
# p = Population(world=w, init_population="Gaussian", num_types=2, compatibility=comp_table, inbreeding_depression=0.0)
comp_table = torch.zeros(3, 3)
comp_table[2, 2] = 1
comp_table[:2, :2] = 1 - torch.eye(2)
p = Population(world=w, init_population="Test2", num_types=3, compatibility=comp_table, inbreeding_depression=0.4,
               seed_rate=10, self_pollen=0.8)
p.make_population()
fig, ax = plt.subplots()
# p.show_population(ax=ax)
for i in range(10):
    p.advance()
    p.show_population(ax=ax, block=False)
