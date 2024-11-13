import numpy as np
import cv2 as cv

img = cv.resize(cv.imread("Aaron_Sorkin_0002.jpg"), (160, 160))
patch = cv.imread("output_best.jpg")
print(patch.shape)
img[80: 120, 80: 120, :] = patch
cv.imwrite("img_patch.png", img)
