from torchvision import transforms
import torch
import cv2
from PIL import Image
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from config_recons import ATTACK_W, RECONS_W


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
    def get_img1(self):
        return self.img1
    def get_location(self):
        return self.location
    def convert_to_3d(self, flattened_patches):
        # print("flattened_patches SHAPE", flattened_patches.shape)
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
        psnr_scores = - self.evaluate_psnr(patchs)
        return adv_scores, psnr_scores
    def __call__(self, patches):
        adv, psnr = self.benchmark(patchs=patches)
        return ATTACK_W * adv + RECONS_W * psnr

if __name__ == '__main__':
    from get_architech import get_model
    img1_path = r'D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Abdel_Nasser_Assidi\Abdel_Nasser_Assidi_0001.jpg'
    img1 = cv2.imread(img1_path)
    img2_path = r'D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Abdel_Nasser_Assidi\Abdel_Nasser_Assidi_0002.jpg'
    img2 = cv2.imread(img2_path)

    model = get_model('restnet_vggface')
    location = (50, 60, 50, 60)
    label = 0
    patch_size = 10
    number_of_individuals = 2
    fitness = Fitness(location, model, img1, img2, label, patch_size)
    test_score = fitness(np.random.randint(0, 10, (number_of_individuals, 3 * patch_size * patch_size)))
    print(test_score)
