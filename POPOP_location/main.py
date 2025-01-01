import argparse
from population import Population
from individual import Individual
from algorithm import GA, NSGAII
from fitness import Fitness
from get_architech import get_model
from dataset import LFW
import cv2
import numpy as np
import os
from torchvision import transforms
import random
from torchvision.utils import save_image
import pickle as pkl

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
    parser.add_argument('--baseline', type=str, default='GA', choices=['GA', 'GA_sequence', 'NSGAII'])
    parser.add_argument('--update_location_iterval', type=int, default=200)
    parser.add_argument('--crossover_type', type=str, choices=['UX', 'Blended'])
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # save
    output_dir = f"{args.baseline}_niter={args.n_iter}_reconsw={args.recons_w}_attackw={args.attack_w}_popsize={args.pop_size}_toursize={args.tourament_size}_patchsize={args.pop_size}_problocationmutate={args.prob_mutate_location}_probpatchmutate={args.prob_mutate_patch}_"
    output_img_dir = os.path.join(output_dir, "img")
    output_pickle_dir = os.path.join(output_dir, "pickle")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_pickle_dir, exist_ok=True)
    
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_crop_margin_5",
               MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
               PAIR_PATH=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/pairs.txt",
               transform=None)
    
    toTensor = transforms.ToTensor()
    
    success_rate = 0
    results = []
    
    for i in range(len(DATA)):
        # if i == 20:
        #     break
        
        random.seed(22520691)
        img1, img2, label = DATA[i]
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
        
        if args.baseline == "GA":
            algo = GA(n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    tourament_size=args.tourament_size,
                    interval_update=args.update_location_iterval,
                    crossover_type=args.crossover_type)
        
        elif args.baseline == "NSGAII":
            algo = NSGAII(n_iter=args.n_iter,
                        population=population,
                        fitness=fitness,
                        tourament_size=args.tourament_size,
                        interval_update=args.update_location_iterval,
                        crossover_type=args.crossover_type)
        
        if args.baseline == 'GA_sequence':
                P, adv_img, adv_score, pnsr_score = algo.solve_sequential()
        elif args.baseline == 'GA':
                print("Using GA")
                P, adv_img, adv_score, pnsr_score = algo.solve()

        elif args.baseline == "NSGAII":
                P, adv_img, adv_score, pnsr_score = algo.solve()
        
        
        # save_image
        save_image(adv_img, os.path.join(output_img_dir, f"{i}.png"))
        # results.append({
        #     "Population": P,
        #     "adv_img": adv_img,
        #     "adv_score": adv_score,
        #     "pnsr_score": pnsr_score})
        
        result = {
                "Population": P,
                "adv_img": adv_img,
                "adv_score": adv_score,
                "pnsr_score": pnsr_score
                }
        
        print("Adv score: ", adv_score)
        if adv_score > 0:
                success_rate += 1
        # break
        output_pickle = os.path.join(output_pickle_dir, f'{i}.pkl')            

        with open(output_pickle, 'wb') as f:
                pkl.dump(result, f)
                        
    print(f"Success rate: {success_rate / 20}")