import cv2
import numpy

def hsv_masking(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    hsv_lower = numpy.array([5,50,50])
    hsv_upper = numpy.array([30,255,255])

    mask = cv2.inRange(hsv,hsv_lower,hsv_upper)

    return mask

def apply_mask(image,mask):
    return cv2.bitwise_and(image,image,mask=mask)