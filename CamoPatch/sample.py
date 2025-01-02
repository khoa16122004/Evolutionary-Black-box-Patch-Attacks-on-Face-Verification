from PIL import Image

class Sample:
    def __init__(self, img1_path, img2_path, transforms):
        img2 = Image.open(img2_path)
        self.img1 = transforms(Image.open(img1_path))
        self.img2 = transforms(Image.open(img1_path))
        