import cv2
import numpy as np
import random
import argparse

from operators.sampling.random_sampling import RandomSampling
from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.selection import RankAndCrowdingSurvival
from problems.patchAttack import PatchFaceAttack
from algorithms import NSGAII
from mtcnn import MTCNN
import os
import pickle as pkl

from config_recons import *

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
         
    return (x_min, x_max, y_min, y_max)
           
                
def parse_args():
    parser = argparse.ArgumentParser(description="Run NSGA-II with PatchFaceAttack")
    parser.add_argument('--n_eval', type=int, required=True, help="Số lần đánh giá cho thuật toán")
    parser.add_argument('--pop_size', type=int, required=True, help="Kích thước quần thể")
    parser.add_argument('--patch_size', type=int, default=20, help="Kích thước patch cho PatchFaceAttack")
    return parser.parse_args()



def popop(args):    
    
    for i in range(len(DATA)):
        mtcnn = MTCNN()
        if i == 100:
            break
        
        for region in ['left_eye', 'right_eye', 'nose']:
            img1, img2, label = DATA[i]
            img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
            img1_np, img2_np = np.array(img1), np.array(img2)
            location = get_landmarks(img1_np, mtcnn, region, args.patch_size)
            
            prob = PatchFaceAttack(args.n_eval, location, MODEL, img1_np, img2_np, label, args.patch_size)
            sampling = RandomSampling(n_sample=args.pop_size, patch_size=args.patch_size)
            crossover = PointCrossover('SE')
            algorithm = NSGAII()
            algorithm.set_hyperparameters(pop_size=args.pop_size, 
                                          sampling=sampling,
                                          crossover=crossover,
                                          mutation=BitStringMutation(),
                                          survival=RankAndCrowdingSurvival(),
                                          debug=True)
            
            algorithm.solve(prob, 22520692)

            F = algorithm.E_Archive_search.F
            X = algorithm.E_Archive_search.X
            
            data = {
                'X': X,
                'F': F
            }
            
            output_dir = f"nsgaII_neval={args.n_eval}_popsize={args.pop_size}"
            output_region_dir= os.path.join(output_dir, region)
            os.makedirs(output_region_dir, exist_ok=True)

            output_file = os.path.join(output_region_dir, f"{i}.pkl")
            with open(output_file, 'wb') as f:
                pkl.dump(data, f)
            



def main():
    args = parse_args()
    popop(args)

if __name__ == "__main__":
    main()
            
            