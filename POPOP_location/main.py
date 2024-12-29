import argparse
from population import Population
from individual import Individual
from algorithm import GA
from fitness import Fitness
from get_architech import get_model
from dataset import LFW
import cv2
import numpy as np
import os
from torchvision import transforms
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patch")
    parser.add_argument('--prob_mutate_location', type=float, default=0.1, help="Probability of mutating the patch location")
    parser.add_argument('--prob_mutate_patch', type=float, default=0.1, help="Probability of mutating the patch itself")
    parser.add_argument('--n_iter', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--tourament_size', type=int, default=3, help="Tournament size for selection")
    parser.add_argument('--recons_w', type=float, default=0.5)
    parser.add_argument('--attack_w', type=float, default=0.5)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_crop_margin_5",
            MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask",
            PAIR_PATH=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/pairs.txt",
            transform=None)
    
    toTensor = transforms.ToTensor()
    random.seed(22520691)
    img1, img2, label = DATA[1]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    
    img1_torch, img2_torch = toTensor(img1), toTensor(img2)
    
    population = Population(pop_size=args.pop_size,
                            patch_size=args.patch_size,
                            img_shape=(160, 160),  # Corrected shape
                            prob_mutate_location=args.prob_mutate_location,
                            prob_mutate_patch=args.prob_mutate_patch)
    
    fitness = Fitness(patch_size=args.patch_size,
                      img1=img1_torch, img2=img2_torch,
                      model=MODEL,  # Use the correct model
                      label=label,
                      recons_w=args.recons_w,
                      attack_w=args.attack_w)
    
    ga = GA(n_iter=args.n_iter,
            population=population,
            fitness=fitness,
            tourament_size=args.tourament_size)
    
    ga.solve()
