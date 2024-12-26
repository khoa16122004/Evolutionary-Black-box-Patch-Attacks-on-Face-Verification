import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image, ImageTk
import pickle as pkl
import cv2 as cv
from config_recons import *

# Configurations
def parse_args():
    parser = argparse.ArgumentParser(description="Pareto Front Visualization")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data.pkl file")
    parser.add_argument('--image_index', type=int, default=30, help="Index of the image to be used")
    parser.add_argument('--patch_size', type=int, default=10, help="Patch size for reconstruction")
    return parser.parse_args()

# Helper Functions
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

def convert_to_3d(flattened_patches, patch_size):
    return np.array([patch.reshape(patch_size, patch_size, 3) for patch in flattened_patches])

def apply_patch_to_image(patch, img, original_location):
    x_min, x_max, y_min, y_max = original_location
    img[y_min:y_max, x_min:x_max, :] = patch.astype('uint8')
    return img

# Main Function
def main():
    args = parse_args()

    with open(args.data_path, "rb") as f:
        data_list = pkl.load(f)

    points = -1 * np.array(data_list['F'])
    imgs = np.array(data_list['X'])
    location = data_list['location']
    # patch_size = data_list['patch_size']
    
    imgs_3d = convert_to_3d(imgs, args.patch_size)

    img1, img2, label = DATA[args.image_index]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    img1_np, img2_np = np.array(img1), np.array(img2)

    img_advs = [
        apply_patch_to_image(imgs_3d[i], img1_np.copy(), location)
        for i in range(imgs_3d.shape[0])
    ]

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
        annot.set_text(f"ADV: {x:.2f}\nFSNR: {y:.2f}")
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
                show_image(img_advs[idx])

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)

    ax.set_xlabel("ADV Fitness")
    ax.set_ylabel("FSNR Fitness")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
