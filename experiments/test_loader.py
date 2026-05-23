import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.image_loader import *

IMAGE_PATH = "../data/raw/0b17e71c-fee6-4b5e-9a7d-8ba44d29215f___Com.G_SpM_FL 8549.JPG"


img = load_image(IMAGE_PATH)
print(img.shape)

rgb = convert_to_rgb(img)
show_image(rgb)

