from model.world import Population, World
import torch

w = World(nutrient_map_type="Rock")
p = Population(world=w)
p.make_population()
p.show_population()
for i in range(15):
    p.advance()
    p.show_population()
