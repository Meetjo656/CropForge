import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.image_loader import load_image
from src.preprocessing.edges import detect_edges
import matplotlib.pyplot as plt

import numpy as np
image = load_image("../data/raw/0b17e71c-fee6-4b5e-9a7d-8ba44d29215f___Com.G_SpM_FL 8549.JPG")

# Check if image is valid before edge detection
if image is None or not isinstance(image, np.ndarray) or image.size == 0:
    print("Error: Image loading failed or returned invalid data.")
    print(f"image type: {type(image)}, dtype: {getattr(image, 'dtype', None)}, shape: {getattr(image, 'shape', None)}")
    edges = None
else:
    edges = detect_edges(image)

# Check if edges is valid before displaying
import numpy as np
if edges is None or not isinstance(edges, np.ndarray) or edges.dtype == object or edges.size == 0:
	print("Error: Edge detection failed or returned invalid data.")
	print(f"edges type: {type(edges)}, dtype: {getattr(edges, 'dtype', None)}, shape: {getattr(edges, 'shape', None)}")
else:
	plt.imshow(edges, cmap='gray')
	plt.title('Edges')
	plt.axis('off')
	plt.show()