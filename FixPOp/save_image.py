from torchvision.utils import save_image
from get_architech import get_model
from dataset import LFW
from fitness import Fitness
from torchvision import transforms
import pickle as pkl
import numpy as np
import os

input_pkl = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle\0.pkl"
log_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal" 
i = 0
output_dir = os.path.join(log_dir, str(i))
os.mkdir(output_dir)

toTensor = transforms.ToTensor()
with open(input_pkl, "rb") as f:
    data_list = pkl.load(f)

MODEL = get_model("restnet_vggface")
DATA = LFW(IMG_DIR=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\lfw_crop_margin_5",
            MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
            PAIR_PATH=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\pairs.txt",
            transform=None)

img1 = DATA[i][0].resize((160, 160))
img1 = toTensor(img1)

fitness = Fitness(patch_size=20,
                        img1=img1,
                        img2=None,
                        model=MODEL,  # Use the correct model
                        label=0,
                        recons_w=None,
                        attack_w=None,
                        fitness_type=None)

Population = data_list['Population']
adv_img = [fitness.apply_patch_to_image(ind.patch, ind.location) for ind in Population]
for j in range(len(adv_img)):
    save_image(adv_img[j], os.path.join(output_dir, f"{j}.png"))
