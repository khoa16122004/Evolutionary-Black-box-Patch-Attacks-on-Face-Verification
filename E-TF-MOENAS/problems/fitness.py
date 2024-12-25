from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


class Fitness:
    def __init__(self,location, model, img1, img2, label, patch_size=20):
        
        self.img1 = img1
        self.img2 = img2
        self.model = model
        self.label = label
        self.location = location
        self.patch_size = patch_size
        
        self.original_patch = self.take_patch_from_image(img1, location)
        img2_torch = transforms.ToTensor()(Image.fromarray(img2)).unsqueeze(0).cuda()
        self.img2_feature = model(img2_torch)
    
    def convert_to_3d(self, flattened_patches):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(self.patch_size, self.patch_size, 3)
        return np.array([patch.reshape(self.patch_size, self.patch_size, 3) for patch in flattened_patches])
    
    def take_patch_from_image(self, img1, location):
        x_min, x_max, y_min, y_max = location
        patch = img1[y_min:y_max, x_min:x_max, :]
        return patch.astype('uint8')
    
    def evaluate_psnr(self, patchs): # np all
        patchs_3d = self.convert_to_3d(patchs)

        r_psnr = np.array([psnr(p[:,:,0], self.original_patch[:,:,0]) for p in patchs_3d])
        g_psnr = np.array([psnr(p[:,:,1], self.original_patch[:,:,1]) for p in patchs_3d])
        b_psnr = np.array([psnr(p[:,:,2], self.original_patch[:,:,2]) for p in patchs_3d])

        psnr_score = (r_psnr + g_psnr + b_psnr) / 3
        
        return psnr_score / 40 

    def apply_patch_to_image(self, patch):
        patchs_3d = self.convert_to_3d(np.array([patch]))
        img_copy = self.img1.copy()
        x_min, x_max, y_min, y_max = self.location
        
        img_copy[y_min : y_max, x_min : x_max, :] = patchs_3d.astype('uint8')
        return img_copy

    def evaluate_adv(self, patchs, threshold=0.5, transform=transforms.ToTensor()):
        patchs_3d = self.convert_to_3d(patchs)
        adv_imgs = [self.apply_patch_to_image(p) for p in patchs_3d]
        
        with torch.no_grad():
            adv_batch = torch.stack([transform(Image.fromarray(img)) for img in adv_imgs]).cuda()

            adv_features = self.model(adv_batch)
            
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            adv_scores = torch.zeros_like(sims).cuda()

            adv_scores = (1 - self.label) * (threshold - sims) + self.label * (sims - threshold)
            
            return adv_scores.cpu().numpy()
        
    def benchmark(self, patchs):
        adv_scores = - self.evaluate_adv(patchs)
        fsnr_scores = - self.evaluate_psnr(patchs)
        return adv_scores, fsnr_scores
