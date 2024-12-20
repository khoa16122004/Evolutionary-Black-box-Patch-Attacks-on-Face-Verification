import string
import random
from skimage.metrics import peak_signal_noise_ratio as psnr 
from PIL import Image, ImageDraw
import cv2
import numpy as np
import imageio 
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from functools import lru_cache
from config_recons import *
import matplotlib.pyplot as plt
import random
from mtcnn import MTCNN
import os
import pickle as pkl

class EarlyStopping:
    def __init__(self, min_improvement=MIN_IMPROVEMENT,
                 target_fitness=TARGET_FITNESS):
        self.min_improvement = min_improvement
        self.target_fitness = target_fitness
        self.best_fitness = -float('inf')
        self.counter = 0
        self.best_generation = 0
        self.best_solution = None
        
    def __call__(self, current_fitness, current_solution, generation):
        if current_fitness >= self.target_fitness:
            print(f"\nTarget fitness {self.target_fitness} achieved at generation {generation}")
            return True
            
        if current_fitness > self.best_fitness + self.min_improvement:
            self.best_fitness = current_fitness
            self.counter = 0
            self.best_generation = generation
            self.best_solution = current_solution
        else:
            self.counter += 1
            
        return False


def get_next_color():
    global COLOR_INDEX
    color = PRECOMPUTED_COLORS[COLOR_INDEX]
    COLOR_INDEX = (COLOR_INDEX + 1) % len(PRECOMPUTED_COLORS)
    return color

def take_patch_from_image(img, original_location):
    img_np = np.array(img)
    x_min, x_max, y_min, y_max = original_location
    patch = img_np[y_min:y_max, x_min:x_max, :]
    return Image.fromarray(patch.astype('uint8'))

@lru_cache(maxsize=128)
def draw_shape(shape_type, x, y, size, color):
    patch = Image.new("RGB", (original_height, original_width), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    
    if shape_type == 'circle':
        draw.ellipse([(y - size, x - size), (y + size, x + size)], fill=color)
    
    elif shape_type == 'square':
        draw.rectangle([(y - size, x - size), (y + size, x + size)], fill=color)
    
    elif shape_type == 'triangle':
        p1 = (y, x - size)
        p2 = (y - size, x + size)
        p3 = (y + size, x + size)
        draw.polygon([p1, p2, p3], fill=color)
    
    elif shape_type == 'ellipse':
        draw.ellipse([(y - size, x - int(size/2)), (y + size, x + int(size/2))], fill=color)
    
    elif shape_type == 'rectangle':
        draw.rectangle([(y - size, x - int(size/2)), (y + size, x + int(size/2))], fill=color)

    return patch

def add_random_shape_to_image(patch, number_of_shapes, original_width, original_height):
    patch_filled = patch.copy()
    draw = ImageDraw.Draw(patch_filled)
    
    shape_types = ['circle', 'square', 'triangle', 'ellipse', 'rectangle']
    
    for _ in range(number_of_shapes):
        x = random.randint(0, original_width - 1)
        y = random.randint(0, original_height - 1)
        size = random.randint(1, 10)
        color = get_next_color()
        
        shape_type = random.choice(shape_types)
        shape_patch = np.array(draw_shape(shape_type, x, y, size, color))
        
        patch_array = np.array(patch_filled)
        mask = shape_patch != 0
        patch_array[mask] = shape_patch[mask]
        patch_filled = Image.fromarray(patch_array)
        
    return patch_filled


def create_random_population(size, orignial_height, original_width):
    return [add_random_shape_to_image(Image.new("RGB", (original_height, original_width), (0,0,0)), 1, original_height, original_width)
            for _ in range(size)]

def apply_patch_to_image(patch, img, original_location):
    img_np = np.array(img)
    patch_np = np.array(patch)
    x_min, x_max, y_min, y_max = original_location
    img_np[y_min:y_max, x_min:x_max, :] = patch_np.astype('uint8')
    return Image.fromarray(img_np)

def evaluate_adv_fitness_batch(adv_imgs, img2s, labels, threshold=0.5, transform=transforms.Compose([transforms.ToTensor()])):
    with torch.no_grad():
        adv_batch = torch.stack([transform(img) for img in adv_imgs]).cuda()
        img2_batch = torch.stack([transform(img) for img in img2s]).cuda()
        
        adv_features = MODEL(adv_batch)
        img2_features = MODEL(img2_batch)
        
        sims = F.cosine_similarity(adv_features, img2_features)
        adv_scores = torch.zeros_like(sims).cuda()

        
        adv_scores = (1 - labels) * (threshold - sims) + labels * (sims - threshold)
        
        
        return adv_scores.cpu().numpy()

def evaluate_psnr_fitness_batch(patches, original_patches):
    patch_arrays = np.stack([np.array(p) for p in patches])
    original_arrays = np.stack([np.array(p) for p in original_patches])
    
    r_psnr = np.array([psnr(p[:,:,0], o[:,:,0]) for p, o in zip(patch_arrays, original_arrays)])
    g_psnr = np.array([psnr(p[:,:,1], o[:,:,1]) for p, o in zip(patch_arrays, original_arrays)])
    b_psnr = np.array([psnr(p[:,:,2], o[:,:,2]) for p, o in zip(patch_arrays, original_arrays)])
    
    psnr_score = (r_psnr + g_psnr + b_psnr) / 3
    return psnr_score / 40

# SO ->, psnr -> , adv ->
def evaluate_fitness_batch(patches, original_patches, original_location, img1s, img2s, labels):
    psnr_fitnesses = evaluate_psnr_fitness_batch(patches, original_patches)
    adv_imgs = [apply_patch_to_image(p, img1, original_location) for p, img1 in zip(patches, img1s)]
    adv_fitnesses = evaluate_adv_fitness_batch(adv_imgs, img2s, labels)
    SO_fitnesses = RECONS_W * psnr_fitnesses + ATTACK_W * adv_fitnesses
    return SO_fitnesses, psnr_fitnesses, adv_fitnesses, adv_imgs 

def images_to_arrays(patch1, patch2):
    return np.array(patch1), np.array(patch2)

def random_horizontal_swap(patch1, patch2, original_height, original_width):
    patch1_arr, patch2_arr = images_to_arrays(patch1, patch2)
    horizontal_random_choice = np.random.choice(original_width,
                                              int(original_width/2),
                                              replace=False)
    patch1_arr[horizontal_random_choice] = patch2_arr[horizontal_random_choice]
    return Image.fromarray(patch1_arr.astype('uint8'))



def random_vertical_swap(patch1, patch2, original_height, original_width):
    patch1_arr, patch2_arr = images_to_arrays(patch1, patch2)
    vertical_random_choice = np.random.choice(original_height, int(original_height / 2), replace=False)
    patch1_arr[:, vertical_random_choice] = patch2_arr[:, vertical_random_choice]
    return Image.fromarray(patch1_arr.astype('uint8'))

def cut_and_merge(patch1, patch2, original_height, original_width):
    patch1_arr, patch2_arr = images_to_arrays(patch1, patch2)
    x_cut = random.randint(0, original_width - 1)
    y_cut = random.randint(0, original_width - 1)
    patch1_arr[x_cut:, y_cut:] = patch2_arr[x_cut:, y_cut:]
    return Image.fromarray(patch1_arr.astype('uint8'))

def single_point_crossover(patch1, patch2, original_height, original_width):
    patch1_arr, patch2_arr = images_to_arrays(patch1, patch2)
    cut_point = random.randint(0, original_height - 1)
    patch1_arr[:cut_point, :] = patch2_arr[:cut_point, :]
    return Image.fromarray(patch1_arr.astype('uint8'))

def two_point_crossover(patch1, patch2, original_height, original_width):
    patch1_arr, patch2_arr = images_to_arrays(patch1, patch2)
    cut_point1 = random.randint(0, original_height // 2)
    cut_point2 = random.randint(cut_point1, original_width)
    patch1_arr[cut_point1:cut_point2, :] = patch2_arr[cut_point1:cut_point2, :]
    return Image.fromarray(patch1_arr.astype('uint8'))

def crossover(patch1, patch2, original_height, original_width):
    crossover_type = random.choice(['horizontal', 'vertical', 'cut_merge', 'single_point', 'two_point'])
    
    if crossover_type == 'horizontal':
        return random_horizontal_swap(patch1, patch2, original_height, original_width)
    elif crossover_type == 'vertical':
        return random_vertical_swap(patch1, patch2, original_height, original_width)
    elif crossover_type == 'cut_merge':
        return cut_and_merge(patch1, patch2, original_height, original_width)
    elif crossover_type == 'single_point':
        return single_point_crossover(patch1, patch2, original_height, original_width)
    elif crossover_type == 'two_point':
        return two_point_crossover(patch1, patch2, original_height, original_width)


def mutate(patch, number_of_times, original_height, original_width):
    return add_random_shape_to_image(patch, number_of_times, original_height, original_width)

def get_parents(local_population, local_fitnesses):
    fitness_sum = sum(np.exp(local_fitnesses))
    fitness_normalized = np.exp(local_fitnesses) / fitness_sum
    
    return [random.choices(local_population, weights=fitness_normalized, k=2) 
            for _ in range(len(local_population))]

def plot_fitness_history(fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title('Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig('fitness_history.png')
    plt.close()
    

def arkive_processing(arkive, new_entry):
    '''
        Domiation definition: x R y if
        x["pnsr_score] >= y["pnsr_score"] and x["adv_score"] >= y["adv_score"] 
    '''
    
    if len(arkive) == 0:
        return [new_entry]
    to_remove = []

    # if exist item dominating new_entry -> no add it to to arkiv
    for i, item in enumerate(arkive):
        if new_entry['psnr_fitnesses'] <= item['psnr_fitnesses'] and new_entry['adv_fitnesses'] <= item['adv_fitnesses']:
            return arkive
    
    # if new_entry dominates item -> remove item from the arkiv
    for i, item in enumerate(arkive):
        if new_entry['psnr_fitnesses'] > item['psnr_fitnesses'] and new_entry['adv_fitnesses'] > item['adv_fitnesses']:
            to_remove.append(i)
            
    # remove with not change the index        
    for idx in reversed(to_remove):
        arkive.pop(idx)
    arkive.append(new_entry)
    return arkive

def final_arkive(merge_arkive):
    merge_arkive.sort(key=lambda x: (x['psnr_fitnesses'], x['adv_fitnesses']), reverse=True)
    final_arkive = []
    for item in merge_arkive:
        final_arkive = arkive_processing(final_arkive, item)
    
    return final_arkive

            
def whole_pipeline(original_patch, img1, img2, label, original_location, original_height, original_width, index):
    
    arkive = [] # for pareto save
    early_stopping = EarlyStopping()
    
    # create original population with params: with len = POPULATION_NUMBER + ELITISM_NUMBER
    population = create_random_population(POPULATION_NUMBER + ELITISM_NUMBER, original_height, original_width)
    
    # create original patch, img1s, img2s
    original_patches = [original_patch] * (POPULATION_NUMBER + ELITISM_NUMBER)
    img1s = [img1] * (POPULATION_NUMBER + ELITISM_NUMBER)
    img2s = [img2] * (POPULATION_NUMBER + ELITISM_NUMBER)
    labels = label
    
    # init
    fitness_history = []
    best_fitness_overall = -float('inf')
    best_patch_overall = None
    
    # pbar
    pbar = tqdm(total=NUMBER_OF_GENERATIONS)
    
    for generation in range(NUMBER_OF_GENERATIONS):
        
        # evaluate population fitness
        fitnesses, psnr_fitnesses, adv_fitnesses, adv_imgs = evaluate_fitness_batch(population, original_patches, original_location, img1s, img2s, labels)        
        best_fitness_idx = np.argmax(fitnesses)
        
        best_fitness = fitnesses[best_fitness_idx] # best fitness score
        best_patch = population[best_fitness_idx] # best patch
        best_adv_imgs = adv_imgs[best_fitness_idx] # assign best patch to img
        
        # frequency scores for arkive saving
        if generation % INTERVAL_ARKIVE == 0:
            # arkive saving
            arkive = arkive_processing(arkive, 
                                {"psnr_fitnesses": psnr_fitnesses[best_fitness_idx], 
                                 "adv_fitnesses": adv_fitnesses[best_fitness_idx], 
                                "adv_img": best_adv_imgs
                                }
                            )
            
        # choosing the best with single objectives
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_adv_imgs = adv_imgs[best_fitness_idx]
            best_patch_overall = best_patch
            
            
        fitness_history.append(best_fitness)
        
        if early_stopping(best_fitness, best_patch, generation):
            print(f"\nStopping early at generation {generation}")
            best_patch = early_stopping.best_solution
            break
            
        top_population_ids = np.argsort(fitnesses)[-ELITISM_NUMBER:]
        parents_list = get_parents(population, fitnesses)
        new_population = []
        
        for i in range(POPULATION_NUMBER):
            new_patch = crossover(parents_list[i][0], parents_list[i][1], original_height, original_width)
            if random.random() < MUTATION_CHANCE:
                new_patch = mutate(new_patch, MUTATION_STRENGTH, original_height, original_width)
            new_population.append(new_patch)
        
        # keep best tinh túy    
        if ELITISM:
            for idx in top_population_ids:
                new_population.append(population[idx])
            
        population = new_population
        pbar.update(1)
    
    pbar.close()


    return Image.fromarray(np.array(best_patch_overall)), arkive


def get_landmarks(img, mtcnn, location=LOCATION, box_size=BOX_SIZE):
    
    """
        Take an list of landmarks
    """
    
    w, h = img.size
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
    
    lmk = lmks[location]

    half_size = box_size // 2
    x_min = max(lmk[0] - half_size, 0)
    y_min = max(lmk[1] - half_size, 0)
    x_max = min(lmk[0] + half_size, w)
    y_max = min(lmk[1] + half_size, h)
    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
     
    original_height, original_width = y_max - y_min, x_max - x_min
    # cv2.imwrite("img_np.png", img_np)
    
    return (x_min, x_max, y_min, y_max), (original_height, original_width)


if __name__ == "__main__":
    output_dir = f"test_loss_{LOCATION}_{RECONS_W}_{ATTACK_W}_{NUMBER_OF_GENERATIONS}"
    os.makedirs(output_dir, exist_ok=True)
    ssr = 0
    
    for i in range(0, len(DATA)):    
        mtcnn = MTCNN()
        if i == 100:
            break
        
        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
        
        
        original_location, (original_height, original_width)  = get_landmarks(img1, mtcnn)
        if not original_location:
            continue
        
        original_patch = take_patch_from_image(img1, original_location)
        
        # evolutionary pipeline
        out_patch, arkiv = whole_pipeline(original_patch, img1, img2, label, 
                                   original_location, original_height, 
                                   original_width, i)



        print(f"\nProcessing pair {i+1}/{len(DATA)}")

        output_adv = apply_patch_to_image(out_patch, img1, original_location)
        
        img1_adv = transforms.ToTensor()(output_adv).cuda()
        img2 = transforms.ToTensor()(img2).cuda()
        img1 = transforms.ToTensor()(img1).cuda()
        
        adv_fea = MODEL(img1_adv.unsqueeze(0))
        img2_fea = MODEL(img2.unsqueeze(0))
        img1_fea = MODEL(img1.unsqueeze(0))
        
        sim = F.cosine_similarity(adv_fea, img2_fea).item()
        sim_0 = F.cosine_similarity(img1_fea, img2_fea).item()
        
        output_path = os.path.join(output_dir, f"adv_{i}_{sim}_{sim_0}_{label}.png")
        output_adv.save(output_path)
        
        output_arkiv_path = output_path.join(output_dir, f'arkiv_{i}.pkl')
        
        with open(output_arkiv_path, "wb") as f:
            pkl.dump(arkiv, f)
