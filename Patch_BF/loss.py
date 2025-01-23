import torch
from torch import nn
import torch.nn.functional as F
from Patch_RS import loss

class LossBF(loss.LossRS):
    def __init__(self, img1: torch.Tensor, img2: torch.Tensor, model: nn.Module):
        super().__init__(img1, img2, model)