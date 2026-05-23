import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.image_loader import load_image
from src.preprocessing.masking import (
    hsv_masking,
    apply_mask
)

from src.severity.severity_calculator import calculate_severity


IMAGE_PATH = "../data/raw/0b17e71c-fee6-4b5e-9a7d-8ba44d29215f___Com.G_SpM_FL 8549.JPG"

image = load_image(IMAGE_PATH)

mask = hsv_masking(image)

masked_output = apply_mask(image, mask)

severity = calculate_severity(mask)

print(f"Disease Severity: {severity:.2f}%")

cv2.imwrite("outputs/mask.jpg", mask)
cv2.imwrite("outputs/masked_output.jpg", masked_output)

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Disease Regions", masked_output)

cv2.waitKey(0)
cv2.destroyAllWindows()