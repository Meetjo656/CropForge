import cv2


def detect_contours(mask):

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_contours(image, contours):

    output = image.copy()

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return output