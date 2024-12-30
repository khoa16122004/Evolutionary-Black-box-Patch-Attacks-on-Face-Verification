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
    parser.add_argument('--baseline', type=str, default='GA', choices=['GA', 'GA_sequence'])
    parser.add_argument('--update_location_iterval', type=int, default=200)
#     parser.add_argument('--output_dir', type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # save
    output_dir = f"{args.baseline}_niter={args.n_iter}_reconsw={args.recons_w}_attackw={args.attack_w}_popsize={args.pop_size}_toursize={args.tourament_size}_patchsize={args.pop_size}_problocationmutate={args.prob_mutate_location}_probpatchmutate={args.prob_mutate_patch}_"
    output_img_dir = os.path.join(output_dir, "img")
    os.makedirs(output_img_dir, exist_ok=True)
    
    MODEL = get_model("restnet_vggface")
    DATA = LFW(IMG_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_crop_margin_5",
               MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
               PAIR_PATH=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/pairs.txt",
               transform=None)
    
    toTensor = transforms.ToTensor()
    
    success_rate = 0
    results = []
    
    for i in range(len(DATA)):
        if i == 1000:
            break
        
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
                tourament_size=args.tourament_size,
                interval_update=args.update_location_iterval)
        
        print("args.baseline")
        if args.baseline == 'GA_sequence':
                adv_img, adv_score, pnsr_score = ga.solve_sequential()
        elif args.baseline == 'GA':
                print("Using GA")
                adv_img, adv_score, pnsr_score = ga.solve()

        # save_image
        print("Adv img", adv_img.shape)
        save_image(adv_img, os.path.join(output_img_dir, f"{i}.png"))
        results.append({
            "adv_img": adv_img,
            "adv_score": adv_score,
            "pnsr_score": pnsr_score})
        
        
        if adv_score > 0:
                success_rate += 1
    
    output_pickle = os.path.join(output_dir, 'result.pkl')            
    with open(output_pickle, 'wb') as f:
            pkl.dump(results, f)
                        
    print(f"Success rate: {success_rate / len(DATA)}")