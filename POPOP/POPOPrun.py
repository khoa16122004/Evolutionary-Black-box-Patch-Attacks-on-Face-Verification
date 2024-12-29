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
from mtcnn import MTCNN


def get_landmarks(img, mtcnn, region, patch_size=20):
    
    """
        Take an list of landmarks
    """
    
    w, h = img.shape[:2]
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    preds = mtcnn.detect_faces(img_np)
    if len(preds) > 1:
        bbs = [item['box'] for item in preds] 
        bb_indx = take_maximum_area_box(bbs)
    elif len(preds) == 1:
        bb_indx = 0
    elif len(preds) == 0:
        return None, (None, None)
    lmks = preds[bb_indx]["keypoints"]
    
    lmk = lmks[region]

    half_size = patch_size // 2
    x_min = max(lmk[0] - half_size, 0)
    y_min = max(lmk[1] - half_size, 0)
    x_max = min(lmk[0] + half_size, w)
    y_max = min(lmk[1] + half_size, h)
    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
         
    return (y_min, y_max, x_min, x_max)

if __name__ == "__main__":
    mtcnn = MTCNN()
    random.seed(22520692)
    img1, img2, label = DATA[1]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    img2.save("img2.png")
    img1_np, img2_np = np.array(img1), np.array(img2)    

    patch_size = 20
    location = get_landmarks(img1_np, mtcnn, 'nose', patch_size=patch_size)
    location = (10, 30, 11, 31)
    
    population_size = 50 # số cá thể trong quần thể
    number_of_shapes = 2 # số hình vẽ ngẫu nhiên
    number_of_generations = 10000 # số thế hệ

    # khởi tạo các hàm 
    fitness_func = Fitness(location=location, 
                           model=MODEL, 
                           img1=img1_np, 
                           img2=img2_np, 
                           label=label, 
                           patch_size=patch_size)
    mutation_func = Mutation(patch_size=patch_size)
    
    crossover_func = Crossover('Random_SE', patch_size)
    
    popop = POPOP(fitness_func=fitness_func, 
                  mutation_func=mutation_func, 
                  crossover_func=crossover_func, 
                  population_size=population_size,
                  patch_size=patch_size,
                  mutation_rate=0.5,
                  visual=True)
                  
    best_patch, best_fitness = popop.evolve(number_of_generations)
    print(f"Best solution: {best_patch}")
    best_patch = best_patch.reshape(patch_size, patch_size, 3)
    save_image_with_patch(img1_np, location, best_patch, "patch_attack_result")    

    print(f"Best fitness: {best_fitness}")
    print(f"fitness: {popop.fitness_values}")
    