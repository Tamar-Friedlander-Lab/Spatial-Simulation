from model.world import Population, World
import time
import matplotlib
from matplotlib import pyplot as plt
import torch
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-population", type=str, default="Gaussian")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--num-types", type=int, default=2)
    parser.add_argument("--inbreeding-depression", type=float, default=0.8)
    parser.add_argument("--self-pollen", type=float, default=0.8)
    parser.add_argument("--max-gen", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=101)
    parser.add_argument("--carry-capacity", type=int, default=20)
    parser.add_argument("--seed-rate", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    fig, ax = plt.subplots()

    w = World(nutrient_map_type="Uniform", size=args.world_size, device=args.device,
              max_nutrient_density=args.carry_capacity)
    if args.num_types == 2:
        comp_table = 1 - torch.eye(2)
        p = Population(world=w, init_population=args.init_population, num_types=args.num_types,
                       compatibility=comp_table, inbreeding_depression=args.inbreeding_depression,
                       self_pollen=args.self_pollen)
    elif args.num_types == 3:
        comp_table = torch.zeros(3, 3)
        comp_table[2, 2] = 1
        comp_table[:2, :2] = 1 - torch.eye(2)
        p = Population(world=w, init_population=args.init_population, num_types=args.num_types,
                       compatibility=comp_table, inbreeding_depression=args.inbreeding_depression,
                       self_pollen=args.self_pollen)
    p.make_population()
    p.show_population(ax=ax)
    for i in range(args.max_gen):
        p.advance()
        p.show_population(ax=ax)


if __name__ == "__main__":
    main()
