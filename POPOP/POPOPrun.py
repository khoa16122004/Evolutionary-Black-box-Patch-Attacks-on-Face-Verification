from .POPOP import POPOP
from .fitness import Fitness
from .mutate import Mutation
from .crossover import Crossover
from get_architech import get_model
import cv2
import numpy as np
import os
from .visualization import save_image_with_patch
from config_recons import *

if __name__ == "__main__":

    random.seed(22520691)
    img1, img2, label = DATA[0]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    img1_np, img2_np = np.array(img1), np.array(img2)    

    location = (50, 70, 50, 70)
    patch_size = 20
    population_size = 50 # số cá thể trong quần thể
    number_of_shapes = 5 # số hình vẽ ngẫu nhiên
    number_of_generations = 1000 # số thế hệ

    # khởi tạo các hàm 
    fitness_func = Fitness(location=location, 
                           model=MODEL, 
                           img1=img1_np, 
                           img2=img2_np, 
                           label=label, 
                           patch_size=patch_size)
    mutation_func = Mutation(patch_size=patch_size)
    
    crossover_func = Crossover()
    
    popop = POPOP(fitness_func=fitness_func, 
                  mutation_func=mutation_func, 
                  crossover_func=crossover_func, 
                  population_size=population_size,
                  patch_size=patch_size,
                  mutation_rate=0.05,
                  visual=True)
                  
    best_patch, best_fitness = popop.evolve(number_of_generations)
    print(f"Best solution: {best_patch}")
    best_patch = best_patch.reshape(patch_size, patch_size, 3)
    save_image_with_patch(img1_np, location, best_patch, "patch_attack_result")    

    print(f"Best fitness: {best_fitness}")
    print(f"fitness: {popop.fitness_values}")