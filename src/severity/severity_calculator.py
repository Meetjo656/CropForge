import cv2
import numpy as np

def calculate_severity(mask):

    inflated_pixels = np.count_nonzero(mask)

    total_pixels = mask.shape[0] * mask.shape[1]

    severity = (inflated_pixels / total_pixels) * 100

    return severity


