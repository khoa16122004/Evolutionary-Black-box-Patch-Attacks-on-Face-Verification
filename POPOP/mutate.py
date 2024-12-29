import numpy as np
import random
import cv2 as cv

class Mutation:
    def __init__(self, patch_size, **kwargs):
        self.patch_size = patch_size

    def convert_to_3d(self, flattened_patches):
        if flattened_patches.shape[0] == 1:
            return flattened_patches[0].reshape(self.patch_size, self.patch_size, 3)
        return np.array([patch.reshape(self.patch_size, self.patch_size, 3) for patch in flattened_patches])
   
    def add_random_shape_to_image(self, patch, number_of_shapes):
        patch_3d = self.convert_to_3d(np.array([patch]))
        image_with_shapes = patch_3d.copy()
        height, width = patch_3d.shape[:2]

        shape_types = ['circle', 'rectangle', 'line', 'ellipse', 'gaussian_noise', 'color_blending']

        for _ in range(number_of_shapes):
            # shape_type = random.choice(shape_types)
            shape_type = 'gaussian_noise'
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
                            
            thickness = random.randint(1, 5)

            if shape_type == 'circle':
                center = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                radius = random.randint(1, min(width, height) // 4)
                cv.circle(image_with_shapes, center, radius, color, thickness)

            elif shape_type == 'rectangle':
                top_left = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                bottom_right = (
                    random.randint(top_left[0], width - 1),
                    random.randint(top_left[1], height - 1)
                )
                cv.rectangle(image_with_shapes, top_left, bottom_right, color, thickness)

            elif shape_type == 'line':
                start_point = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                end_point = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                cv.line(image_with_shapes, start_point, end_point, color, thickness)

            elif shape_type == 'ellipse':
                center = (
                    random.randint(0, width - 1),
                    random.randint(0, height - 1)
                )
                axes = (
                    random.randint(1, width // 4),
                    random.randint(1, height // 4)
                )
                angle = random.randint(0, 360)
                start_angle = 0
                end_angle = 360
                cv.ellipse(image_with_shapes, center, axes, angle, start_angle, end_angle, color, thickness)

            elif shape_type == 'gaussian_noise':
                noise = np.random.normal(0, 5, image_with_shapes.shape)  # Mean=0, Stddev=5
                noise = noise.astype(np.float32)  # Tránh tràn giá trị
                image_with_shapes = np.clip(image_with_shapes + 0.01 * noise, 0, 255).astype(np.uint8)

            # elif shape_type == 'color_blending':
            #     blend_color = np.zeros_like(image_with_shapes, dtype=np.uint8)
            #     blend_color[:, :, 0] = color[0]
            #     blend_color[:, :, 1] = color[1]
            #     blend_color[:, :, 2] = color[2]
            #     alpha = random.uniform(0.1, 0.5)  # Tỉ lệ pha trộn
            #     image_with_shapes = np.clip(image_with_shapes * (1 - alpha) + blend_color * alpha, 0, 255).astype(np.uint8)

        return image_with_shapes
        
    def __call__(self, patch, number_of_shapes=2):
        patch_final = self.add_random_shape_to_image(patch, number_of_shapes)
        return patch_final.flatten()

if __name__ == "__main__":
    number_of_individuals = 1
    patch_size = 3
    mutation = Mutation(patch_size=patch_size)
    patch = np.random.randint(0, 10, (3 * patch_size * patch_size))
    print("patch: ", patch.shape)
    patch_mutated = mutation(patch)
    print("patch mutated: ", patch_mutated.shape)
    print("patch mutated: ", patch_mutated.flatten().shape)