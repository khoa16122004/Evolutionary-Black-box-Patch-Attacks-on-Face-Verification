import os
import cv2

def save_image_with_patch(img, location, patch, outdir):
    # print(type(patch))
    xmin, xmax, ymin, ymax = location
    img[xmin:xmax, ymin:ymax, :] = patch
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, "patched_image.png")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
