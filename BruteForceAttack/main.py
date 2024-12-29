import argparse
from config_recons import *
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.utils import save_image

def cal_sims(img1_torch, img2_torch, MODEL):
    img1_feature = MODEL(img1_torch)
    img2_feature = MODEL(img2_torch)

    sim = F.cosine_similarity(img1_feature, img2_feature)
    return sim
    
def isAttacked(sim, label, threshold=0.5):
    if (sim < threshold and label == 0) or (sim > threshold and label == 1):
        return True
    return False

def random_Attack(img1_torch, img2_torch, label, patch_size, index, std, log_file):
    with open(log_file, "w") as f:
        for i in range(img1_torch.shape[2] - patch_size):  # Height
            for j in range(img1_torch.shape[3] - patch_size):  # Width
                img1_copy = img1_torch.clone()
                img1_copy[0, :, i:i+patch_size, j:j+patch_size] = torch.randn((img1_torch.shape[1], patch_size, patch_size), device="cuda") * std
                save_image(img1_copy, "process.png")
                
                sim = cal_sims(img1_copy, img2_torch, MODEL)
                print(sim)
                if isAttacked(sim, label):
                    f.write(f"[Attack successfully]: index: {index}, label: {label}, location: {(i, j)}\n")

def main():
    parser = argparse.ArgumentParser(description="Random Attack Simulation")
    parser.add_argument('--patch_size', type=int, required=True, help="Size of the patch for the attack")
    parser.add_argument('--std', type=float, required=True, help="Standard deviation of the Gaussian noise")
    args = parser.parse_args()

    patch_size = args.patch_size
    std = args.std
    log_file = f"Random_Attack_log_patchsize={args.patch_size}_std={args.std}.txt"

    toTensor = ToTensor()

    for i in range(len(DATA)):
        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
        img1_torch, img2_torch = toTensor(img1).unsqueeze(0).cuda(), toTensor(img2).unsqueeze(0).cuda()

        random_Attack(img1_torch, img2_torch, label, patch_size, i, std, log_file)

if __name__ == '__main__':
    main()
