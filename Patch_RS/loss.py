import torch
from torch import nn
import torch.nn.functional as F

class LossRS:
    def __init__(self, img1:torch.Tensor, img2: torch.Tensor, model: nn.Module) -> None:
        self.img1 = img1.cuda()
        self.img2_feature = model(img2.cuda().unsqueeze(0))
        self.model = model.eval()
    def evaluate_adv(self, adv_img: torch.Tensor):
        
        with torch.no_grad():
            adv_img = adv_img.cuda().unsqueeze(0)
            adv_features = self.model(adv_img)
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            adv_scores = 0.5 - sims
            
            return adv_scores
            
    def evaluate_psnr(self, adv_img: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(adv_img.cuda(), self.img1.expand_as(adv_img).cuda(), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1) 
        psnr_scores = torch.log10(1 / (mse + 1e-8))
        
        return (sum(psnr_scores)/3 )/ 10    
    
    def __call__(self, adv_img: torch.Tensor) -> torch.Tensor:
        adv_scores = self.evaluate_adv(adv_img)
        psnr_scores = self.evaluate_psnr(adv_img)

        return psnr_scores, adv_scores, adv_scores > 0
        