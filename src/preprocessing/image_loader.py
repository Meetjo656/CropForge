import cv2
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {path}")
    
    return img

def resize_image(img,length,height):
    return cv2.resize(img,(length,height))

def convert_to_rgb(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
def show_image(image):
    import matplotlib.pyplot as plt
    # If image is BGR (OpenCV default), convert to RGB for matplotlib
    if len(image.shape) == 3 and image.shape[2] == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    