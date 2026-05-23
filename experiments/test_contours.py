import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.image_loader import load_image
from src.preprocessing.masking import hsv_masking
from src.preprocessing.morphology import clean_mask
from src.preprocessing.contours import (
    detect_contours,
    draw_contours
)

image = load_image("../data/raw/0b17e71c-fee6-4b5e-9a7d-8ba44d29215f___Com.G_SpM_FL 8549.JPG")
mask = hsv_masking(image)

cleaned_mask = clean_mask(mask)

contours = detect_contours(cleaned_mask)

output = draw_contours(image, contours)

output = draw_contours(image, contours)

print(f"Detected Lesions: {len(contours)}")

cv2.imshow("Contours", output)
cv2.imshow("Mask", mask)
cv2.imshow("Cleaned Mask", cleaned_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()