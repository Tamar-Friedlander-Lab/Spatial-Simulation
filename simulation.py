from world import Population, World
import torch

w = World()
p = Population(world=w)
p.MakePopulation()
p.ShowPopulation(out_file="tests/init_population.png")
print(w.max_nutrient_density)
print(torch.std(w.nutrient_map))
