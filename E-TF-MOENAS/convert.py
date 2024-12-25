import pickle as pkl
import numpy as np
from config_recons import *
import cv2 as cv

def convert_to_3d(flattened_patches, patch_size=10):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(patch_size, patch_size, 3)
        return np.array([patch.reshape(patch_size, patch_size, 3) for patch in flattened_patches])

with open("data_té.pkl", "rb") as f:
    load_data = pkl.load(f)

def apply_patch_to_image(patch, img, original_location):
    x_min, x_max, y_min, y_max = original_location
    img[y_min:y_max, x_min:x_max, :] = patch.astype('uint8')
    return img


X_loaded = np.array(load_data['X'])
F_loaded = np.array(load_data['F'])

X_3d = convert_to_3d(X_loaded)

img1, img2, label = DATA[0]
img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))

img1_np, img2_np = np.array(img1), np.array(img2)
location, (w, h) = (90, 100, 100, 110), (10, 10)

img_adv = apply_patch_to_image(X_3d[0], img1_np, location)
img_adv_bgr = cv.cvtColor(img_adv, cv.COLOR_RGB2BGR)

cv.imwrite("test.png", img_adv_bgr)
print(X_3d.shape)