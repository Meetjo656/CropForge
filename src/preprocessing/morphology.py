import cv2
import numpy as np


def clean_mask(mask):

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closing