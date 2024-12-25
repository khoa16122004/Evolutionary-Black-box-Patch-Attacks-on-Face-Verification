import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image, ImageTk
import pickle as pkl
from config_recons import *
import cv2 as cv 

def pareto_front(points):
    pareto_points = points[np.argsort(points[:, 0])]
    return pareto_points


def show_image(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))

    window = tk.Toplevel()
    window.title("Adversarial Image")

    img_tk = ImageTk.PhotoImage(img_pil)

    label = tk.Label(window, image=img_tk)
    label.image = img_tk  
    label.pack()

    window.mainloop()
def convert_to_3d(flattened_patches, patch_size=10):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(patch_size, patch_size, 3)
        return np.array([patch.reshape(patch_size, patch_size, 3) for patch in flattened_patches])

def apply_patch_to_image(patch, img, original_location):
    x_min, x_max, y_min, y_max = original_location
    img[y_min:y_max, x_min:x_max, :] = patch.astype('uint8')
    return img
with open(r"D:\Path-Recontruction-with-Evolution-Strategy\E-TF-MOENAS\data.pkl", "rb") as f:
    data_list = pkl.load(f)

# points = np.array([[d["psnr_fitnesses"], d["adv_fitnesses"]] for d in data_list])
points = -1 * np.array(data_list['F'])
imgs = np.array(data_list['X'])
print("Diff: ", (imgs[0] - imgs[1]).sum())


imgs_3d = convert_to_3d(imgs, 10)
print("Diff: ", (imgs_3d[0] - imgs_3d[1]).sum())


img1, img2, label = DATA[30]
img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))

img1_np, img2_np = np.array(img1), np.array(img2)
location, (w, h) = (50, 60, 50, 60), (10, 10)


img_advs = [apply_patch_to_image(imgs_3d[i], img1_np.copy(), location) for i in range(imgs_3d.shape[0])]
print("Diff: ", (img_advs[0] - img_advs[1]).sum())

# cv.imwrite("0.png", cv.cvtColor(img_advs[0], cv.COLOR_BGR2RGB))
# cv.imwrite("1.png", cv.cvtColor(img_advs[1], cv.COLOR_BGR2RGB))
# cv.imwrite("2.png", cv.cvtColor(img_advs[20], cv.COLOR_BGR2RGB))

pareto_points = pareto_front(points)

fig, ax = plt.subplots()
scatter = ax.scatter(points[:, 0], points[:, 1], label="Data Points", picker=True, s=10)
ax.plot(pareto_points[:, 0], pareto_points[:, 1], color='red', label="Pareto Front")

cursor = Cursor(ax, useblit=True, color='green', linewidth=1)

annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    index = ind["ind"][0]
    x, y = points[index]
    annot.xy = (x, y)
    annot.set_text(f"ADV: {x:.2f}\FSNR: {y:.2f}")
    annot.get_bbox_patch().set_alpha(0.4)

def on_hover(event):
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            annot.set_visible(False)
            fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        idx = np.argmin(distances)
        print(idx)
        if distances[idx] < 0.5:  # Adjust sensitivity
            show_image(img_advs[idx])

fig.canvas.mpl_connect('motion_notify_event', on_hover)
fig.canvas.mpl_connect('button_press_event', on_click)

ax.set_xlabel("ADV Fitness")
ax.set_ylabel("FSNR Fitness")
ax.legend()
plt.show()
