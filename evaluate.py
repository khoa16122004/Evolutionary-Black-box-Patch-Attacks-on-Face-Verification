from config_recons import MODEL
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch

img1_path = "D:\Path-Recontruction-with-Evolution-Strategy\patch_attack_result\patched_image.png"
img2_path = "D:\Path-Recontruction-with-Evolution-Strategy\img2.png"

img1 = Image.open(img1_path).convert('RGB').resize((160, 160))
img2 = Image.open(img2_path).convert('RGB').resize((160, 160))

transform = transforms.ToTensor()
img1_torch = transform(img1).unsqueeze(0).cuda()
img1_torch[:, :, 91:101, 75:85] = 0
img2_torch = transform(img2).unsqueeze(0).cuda()

with torch.no_grad():
    img1_features = MODEL(img1_torch)
    img2_features = MODEL(img2_torch)

    sims = F.cosine_similarity(img1_features, img2_features)
    print(sims)