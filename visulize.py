import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_adv_scatter(psnr_fitnesses, adv_fitnesses, adv_imgs, best_fitness_idx=None):
    plt.figure(figsize=(25, 25))
    
    scatter = plt.scatter(psnr_fitnesses, adv_fitnesses, 
                          c=adv_fitnesses, 
                          cmap='viridis', 
                          s=100)
    
    sorted_indices = np.argsort(psnr_fitnesses)
    sorted_psnr = np.array(psnr_fitnesses)[sorted_indices]
    sorted_adv = np.array(adv_fitnesses)[sorted_indices]
    plt.plot(sorted_psnr, sorted_adv, color='blue', linestyle='--', label='Pareto Path')
    
    
    
    if best_fitness_idx is not None:
        plt.scatter(psnr_fitnesses[best_fitness_idx], 
                    adv_fitnesses[best_fitness_idx], 
                    color='red', 
                    s=200, 
                    edgecolors='black', 
                    label='Best Point')
    
    plt.xlabel('PSNR', fontsize=12)
    plt.ylabel('Adversarial Fitness', fontsize=12)
    plt.title('PSNR vs Adversarial Fitness', fontsize=14)
    
    plt.colorbar(scatter, label='Adversarial Fitness')
    
    def add_image_annotation(x, y, img):
        imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (x, y), 
                            xycoords='data', 
                            boxcoords="offset points",
                            bboxprops=dict(alpha=0.5),
                            pad=0.3)
        plt.gca().add_artist(ab)
    
    for i, (x, y) in enumerate(zip(psnr_fitnesses, adv_fitnesses)):
        add_image_annotation(x, y, adv_imgs[i])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    plt.show()

def main():
    with open(r"D:\Path-Recontruction-with-Evolution-Strategy\test_loss_nose_0.6_0.4_20000\arkiv_0.pkl", "rb") as f:
        data_list = pkl.load(f)
    
    psnr_fitnesses = [item['psnr_fitnesses'] for item in data_list]
    adv_fitnesses = [item['adv_fitnesses'] for item in data_list]
    adv_imgs = [item['adv_img'] for item in data_list]
    
    best_fitness_idx = np.argmax(adv_fitnesses)
    
    plot_adv_scatter(psnr_fitnesses, adv_fitnesses, adv_imgs, best_fitness_idx)

if __name__ == "__main__":
    main()