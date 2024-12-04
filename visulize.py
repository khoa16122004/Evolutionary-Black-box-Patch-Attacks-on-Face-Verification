import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_adv_scatter(psnr_fitnesses, adv_fitnesses, adv_imgs, best_fitness_idx=None):
    """
    Create a scatter plot of PSNR vs Adversarial Fitness with image previews.
    
    Parameters:
    - psnr_fitnesses: List of PSNR values
    - adv_fitnesses: List of adversarial fitness values
    - adv_imgs: List of adversarial images
    - best_fitness_idx: Index of the best performing point (optional)
    """
    # Create the plot
    plt.figure(figsize=(50, 50))
    
    # Scatter plot
    scatter = plt.scatter(psnr_fitnesses, adv_fitnesses, 
                          c=adv_fitnesses, 
                          cmap='viridis', 
                          s=100)
    
    # Highlight best point if specified
    if best_fitness_idx is not None:
        plt.scatter(psnr_fitnesses[best_fitness_idx], 
                    adv_fitnesses[best_fitness_idx], 
                    color='red', 
                    s=200, 
                    edgecolors='black', 
                    label='Best Point')
    
    # Labeling
    plt.xlabel('PSNR', fontsize=12)
    plt.ylabel('Adversarial Fitness', fontsize=12)
    plt.title('PSNR vs Adversarial Fitness', fontsize=14)
    
    # Colorbar
    plt.colorbar(scatter, label='Adversarial Fitness')
    
    # Add image previews
    def add_image_annotation(x, y, img):
        imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (x, y), 
                            xycoords='data', 
                            boxcoords="offset points",
                            bboxprops=dict(alpha=0.5),
                            pad=0.3)
        plt.gca().add_artist(ab)
    
    # Add image previews for all points
    for i, (x, y) in enumerate(zip(psnr_fitnesses, adv_fitnesses)):
        add_image_annotation(x, y, adv_imgs[i])
    
    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Tight layout to prevent cutting off labels
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def main():
    # Load data from pickle file
    with open("arkiv.pkl", "rb") as f:
        data_list = pkl.load(f)
    
    # Extract values from the list of dictionaries
    psnr_fitnesses = [item['psnr_fitnesses'] for item in data_list]
    adv_fitnesses = [item['adv_fitnesses'] for item in data_list]
    adv_imgs = [item['adv_img'] for item in data_list]
    
    # Find best fitness index
    best_fitness_idx = np.argmax(adv_fitnesses)
    
    # Plot
    plot_adv_scatter(psnr_fitnesses, adv_fitnesses, adv_imgs, best_fitness_idx)

if __name__ == "__main__":
    main()