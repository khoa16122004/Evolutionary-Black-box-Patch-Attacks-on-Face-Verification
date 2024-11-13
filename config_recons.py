from dataset import LFW
from get_architech import get_model
from torchvision import transforms
import random

MODEL = get_model("restnet_vggface")
DATA = LFW(IMG_DIR="lfw_dataset/lfw_crop_margin_5",
           MASK_DIR="lfw_dataset/lfw_lips_mask",
           PAIR_PATH="lfw_dataset/pairs.txt",
           transform=None)

PATIENCE = 10
# ORIGINAL_LOCATION = (90, 110, 80, 100)
# ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 20, 20
NUMBER_OF_GENERATIONS = 50000
POPULATION_NUMBER = 44
MUTATION_CHANCE = 0.2
MUTATION_STRENGTH = 1
ELITISM = True
ELITISM_NUMBER = 6
STARTING_SHAPE_NUMBER = 6
PRINT_EVERY_GEN = 25
SAVE_FRAME_FOR_GIF_EVERY = 100
ATTACK_W = 9
RECONS_W = 1.2

MIN_IMPROVEMENT = 10
TARGET_FITNESS = 40

PRECOMPUTED_COLORS = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
                      for _ in range(1000)]
COLOR_INDEX = 0