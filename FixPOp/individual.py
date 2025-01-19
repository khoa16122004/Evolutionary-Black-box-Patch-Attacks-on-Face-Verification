import random
import torch
from torchvision.utils import save_image

class Individual:
    def __init__(self, patch_size: int, img_shape: tuple[int, int], prob_mutate_patch: float, prob_mutate_location: float) -> None:
        """
        Initialize an individual with a random patch and location.
        """
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location
        self.rank = None
        self.crowding = None
        
        self._random_location()
        self._random_patch()
        
        self.psnr_score = None
        self.adv_score = None

    def _random_location(self) -> None:
        """
        Generates a random location (x_min, x_max, y_min, y_max) within image bounds.
        """
        x_min = random.randint(0, self.img_shape[0] - self.patch_size)
        y_min = random.randint(0, self.img_shape[1] - self.patch_size)
        x_max, y_max = x_min + self.patch_size, y_min + self.patch_size
        
        self.location = (x_min, x_max, y_min, y_max)
    
    def _random_patch(self) -> None:
        """
        Generates a random patch of shape (3, patch_size, patch_size).
        """
        self.patch = torch.rand(3, self.patch_size, self.patch_size).cuda()
    
    def mutate(self) -> None:
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  # Add rectangle
            self._add_rectangle()
        
        if random.random() < self.prob_mutate_location:
            self._random_location()
 
    def mutate_location(self) -> None:
        """
        Apply a mutation location to the individual
        """

        self._random_location()
    
    def mutate_content(self) -> None:
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  
            self._add_rectangle()
    
    def _add_rectangle(self) -> None:
        """
        Add a rectangle to the patch.
        """
        x_min = random.randint(0, self.patch_size - 1)
        y_min = random.randint(0, self.patch_size - 1)
        width = random.randint(2, 5)
        color = torch.rand(3).cuda()  # Random RGB color

        self.patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)
    

    def crossover_UX(self, parent2: 'Individual') -> tuple['Individual', 'Individual']:
        """
        Perform crossover with another individual to produce two offspring.

        :param parent2: Another Individual object.
        :return: Two new Individual objects.
        """
        offstring1_patch = self.patch.clone()
        offstring2_patch = parent2.patch.clone()

        offstring1 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)
        offstring2 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)

        cut_point = random.randint(0, self.patch_size)
        offstring1_patch[:, :cut_point, :] = parent2.patch[:, :cut_point, :]
        offstring2_patch[:, :cut_point, :] = self.patch[:, :cut_point, :]

        if random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location 
      
        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch

        return offstring1, offstring2
    
    def crossover_blended(self, parent2: 'Individual', alpha=0.5) -> tuple['Individual', 'Individual']:
        """
        using crossover_blended
        o1 = alpha * p1 + (1 - alpha) * p2
        o2 = alpha *p2 + (1 - alpha) * p1
        """    
        offstring1_patch = alpha * self.patch + (1 - alpha) * parent2.patch
        offstring2_patch = alpha * parent2.patch + (1 - alpha) * self.patch

        offstring1 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)
        offstring2 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)

        if random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location
        
        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch

        return offstring1, offstring2
        
        
