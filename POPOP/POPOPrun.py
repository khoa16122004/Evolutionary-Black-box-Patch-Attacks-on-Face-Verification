from .POPOP import POPOP
from .fitness import Fitness
from .mutate import Mutation
from .crossover import Crossover
from get_architech import get_model
import cv2
import numpy as np
import os
from .visualization import save_image_with_patch

if __name__ == "__main__":

    # khởi tạo các tham số
    img1_path = r'D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Abdel_Nasser_Assidi\Abdel_Nasser_Assidi_0001.jpg'
    img1 = cv2.imread(img1_path)
    img2_path = r'D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Abdel_Nasser_Assidi\Abdel_Nasser_Assidi_0002.jpg'
    img2 = cv2.imread(img2_path)
    model = get_model('restnet_vggface')
    location = (50, 60, 50, 60)
    label = 0
    patch_size = 10
    population_size = 40 # số cá thể trong quần thể
    number_of_shapes = 2 # số hình vẽ ngẫu nhiên
    number_of_generations = 1000 # số thế hệ

    # khởi tạo các hàm 
    fitness_func = Fitness(location=location, 
                           model=model, 
                           img1=img1, 
                           img2=img2, 
                           label=label, 
                           patch_size=patch_size)
    mutation_func = Mutation(patch_size=patch_size)
    crossover_func = Crossover()
    popop = POPOP(fitness_func=fitness_func, 
                  mutation_func=mutation_func, 
                  crossover_func=crossover_func, 
                  population_size=population_size,
                  patch_size=patch_size,
                  visual=True)
                  
    best_patch, best_fitness = popop.evolve(number_of_generations)
    print(f"Best solution: {best_patch}")
    best_patch = best_patch.reshape(patch_size, patch_size, 3)
    save_image_with_patch(img1, location, best_patch, "patch_attack_result")    

    print(f"Best fitness: {best_fitness}")
    print(f"fitness: {popop.fitness_values}")