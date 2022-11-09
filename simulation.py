from model.world import Population, World
import torch

w = World()
p = Population(world=w)
p.make_population()
p.show_population(out_file="tests/init_population.png")
print(w.max_nutrient_density)
print(torch.std(w.nutrient_map))
