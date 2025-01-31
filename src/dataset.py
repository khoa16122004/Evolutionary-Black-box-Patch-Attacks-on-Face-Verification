import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import numpy as np
import json
from PIL import Image
from torchvision import transforms  
from torchvision.datasets.utils import download_and_extract_archive

class LFW(Dataset):        
    def __init__(self, 
                 IMG_DIR: str,
                 MASK_DIR: str,
                 PAIR_PATH: str,
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize((160, 160)),
                                               ]),
                 ):
        with open(PAIR_PATH, "r") as f:
            f.readline()
            lines = [line.strip().split("\t") for line in f.readlines()]
         
        self.lines = lines
        self.IMG_DIR = IMG_DIR
        self.MASK_DIR = MASK_DIR
        self.transform = transform
         
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        if len(line) == 3:
            first_iden_name, first_id, second_id = line
            second_iden_name = first_iden_name 
            label = 0           
        elif len(line) == 4:
            first_iden_name, first_id, second_iden_name, second_id = line
            label = 1
        
        first_name = f"{first_iden_name}_{first_id.zfill(4)}.jpg" 
        first_path = os.path.join(self.IMG_DIR, first_iden_name, first_name)
        first_mask_path = first_path.replace(self.IMG_DIR, self.MASK_DIR)
        
        
        second_name = f"{second_iden_name}_{second_id.zfill(4)}.jpg"
        second_path =  os.path.join(self.IMG_DIR, second_iden_name, second_name)
        second_mask_path = second_path.replace(self.IMG_DIR, self.MASK_DIR)
        
        
        first_image = Image.open(first_path).convert("RGB")
        second_image = Image.open(second_path).convert("RGB")
        
        
        if self.transform:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image) 

                
        return first_image, second_image, label
    
    
    
class LFW_EVALUATION(Dataset):
    
    def preprocessing_image(self, img, lmk):
        aimg = norm_crop(img, lmk, img.shape[0])
        return aimg
    
    def __init__(self, 
                 IMG_DIR: str,
                 PAIR_PATH: str,
                 LMK_PATH: str,
                ):
        with open(PAIR_PATH, "r") as f:
            f.readline()
            lines = [line.strip().split("\t") for line in f.readlines()]
        
        with open(LMK_PATH, "r") as f:
            lmks_dict = json.load(f)
        
        self.lmks_dict = lmks_dict
        self.lines = lines
        self.IMG_DIR = IMG_DIR
        
         
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        if len(line) == 3:
            first_iden_name, first_id, second_id = line
            second_iden_name = first_iden_name 
            label = 0           
        elif len(line) == 4:
            first_iden_name, first_id, second_iden_name, second_id = line
            label = 1
        
        first_name = f"{first_iden_name}_{first_id.zfill(4)}.jpg" 
        first_path = os.path.join(self.IMG_DIR, first_iden_name, first_name)
        
        second_name = f"{second_iden_name}_{second_id.zfill(4)}.jpg"
        second_path =  os.path.join(self.IMG_DIR, second_iden_name, second_name)
                
        first_image = cv.cvtColor(cv.imread(first_path), cv.COLOR_BGR2RGB)
        second_image = cv.cvtColor(cv.imread(second_path), cv.COLOR_BGR2RGB)
        
        first_image = self.preproccessing(first_image, self.lmks_dict[first_name])
        second_image = self.preproccessing(second_image, self.lmks_dict[second_name])
        
        return first_image, second_image, label
        


class simpleLFW(Dataset):
    def __init__(self,
                 imgDir: str,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize((160, 160)),
                 ])):
        
        imgPaths = []
        for indenty in os.listdir(imgDir):
            identyPath = os.path.join(imgDir, indenty)
            for fileName in os.listdir(identyPath):
                filePath = os.path.join(identyPath, fileName)
                imgPaths.append(filePath)
        
        self.imgPaths = imgPaths
        self.transform = transform
    
    def __len__(self):
        return len(self.imgPaths)
    def __getitem__(self, indx):
        imgPath = self.imgPaths[indx]
        img = Image.open(imgPath).convert("RGB")
        img = self.transform(img)
        return img