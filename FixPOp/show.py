import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image, ImageTk
import pickle as pkl
from get_architech import get_model
from dataset import LFW
from fitness import Fitness
from torchvision import transforms
import os
import cv2 as cv

toTensor = transforms.ToTensor()
i = 0
with open(r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\pickle\0.pkl", "rb") as f:
    data_list = pkl.load(f)

# NSGAII
img_dir = r"D:\Path-Recontruction-with-Evolution-Strategy\FixPOp\arkiv_NSGAII_niter=10000_label=0_reconsw=0.0_attackw=1.0_popsize=80_toursize=4_patchsize=80_problocationmutate=0.3_probpatchmutate=0.5_fitnesstype=normal\0"
adv_img = []
for file_name in os.listdir(img_dir):
    file_path = os.path.join(img_dir, file_name)
    # img = cv.imread(file_path)
    # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_rgb = Image.open(file_path).convert("RGB")
    adv_img.append(img_rgb)

# GA - normal

MODEL = get_model("restnet_vggface")
DATA = LFW(IMG_DIR=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\lfw_crop_margin_5",
            MASK_DIR=r"D:/Path-Recontruction-with-Evolution-Strategy/lfw_dataset/lfw_lips_mask", 
            PAIR_PATH=r"D:\Path-Recontruction-with-Evolution-Strategy\lfw_dataset\pairs.txt",
            transform=None)

img1 = DATA[i][0].resize((160, 160))
img1 = toTensor(img1)

fitness = Fitness(patch_size=20,
                        img1=img1,
                        img2=None,
                        model=MODEL,  # Use the correct model
                        label=0,
                        recons_w=None,
                        attack_w=None,
                        fitness_type=None)

Population = data_list['Population']
points = np.array([[ind.psnr_score.item(), ind.adv_score.item()] for ind in Population])
# points = np.array([[d["psnr_fitnesses"], d["adv_fitnesses"]] for d in data_list])

def pareto_front(points):
    pareto_points = points[np.argsort(points[:, 0])]
    return pareto_points

pareto_points = pareto_front(points)

def show_image(img):
    window = tk.Toplevel()
    window.title("Adversarial Image")
    
    img_tk = ImageTk.PhotoImage(img)
    
    label = tk.Label(window, image=img_tk)
    label.image = img_tk 
    label.pack()
    window.mainloop()

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
    annot.set_text(f"PSNR: {x:.2f}\nADV: {y:.2f}")
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
        if distances[idx] < 0.5:  # Adjust sensitivity
            
            show_image(adv_img[idx])

fig.canvas.mpl_connect('motion_notify_event', on_hover)
fig.canvas.mpl_connect('button_press_event', on_click)

ax.set_xlabel("PSNR Fitness")
ax.set_ylabel("ADV Fitness")
ax.legend()
plt.show()
