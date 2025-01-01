import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image, ImageTk
import pickle as pkl

with open(r"D:\Path-Recontruction-with-Evolution-Strategy\new_0.5_0.5_10000_22520691\arkiv_9.pkl", "rb") as f:
    data_list = pkl.load(f)

points = np.array([[d["psnr_fitnesses"], d["adv_fitnesses"]] for d in data_list])

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
            show_image(data_list[idx]["adv_img"])

fig.canvas.mpl_connect('motion_notify_event', on_hover)
fig.canvas.mpl_connect('button_press_event', on_click)

ax.set_xlabel("PSNR Fitness")
ax.set_ylabel("ADV Fitness")
ax.legend()
plt.show()
