import cv2
import numpy as np
from ultralytics import SAM

class SAM2Segmenter:
    def __init__(self, model_path="sam2.1_t.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = SAM(self.model_path)

    def predict(self, image, boxes):
        """
        Run SAM2 segmentation using bounding boxes as prompts.
        image: numpy array (BGR image) or path
        boxes: ultralytics.engine.results.Boxes object or list of boxes
        """
        if self.model is None:
            self.load_model()
            
        # Extract xyxy boxes if it's a YOLO boxes object
        bboxes = None
        if hasattr(boxes, 'xyxy'):
            bboxes = boxes.xyxy.cpu().numpy().tolist()
        else:
            bboxes = boxes
            
        results = self.model.predict(source=image, bboxes=bboxes, verbose=False)[0]
        return results.masks

    def overlay(self, image, masks):
        """
        Overlay the generated masks on the image.
        Returns the overlay image.
        """
        if masks is None or len(masks.data) == 0:
            return image.copy()
            
        overlay_img = image.copy()
        
        # Merge all masks into one binary mask
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for mask_tensor in masks.data:
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.logical_or(combined_mask, mask_resized).astype(np.uint8)
            
        # Apply color overlay (e.g., green for mask)
        color = np.array([0, 255, 0], dtype=np.uint8)
        alpha = 0.5
        
        colored_mask = np.zeros_like(image)
        colored_mask[combined_mask == 1] = color
        
        cv2.addWeighted(colored_mask, alpha, overlay_img, 1 - alpha, 0, overlay_img)
        
        return overlay_img

    def save(self, image, path):
        """
        Save the image to the specified path.
        """
        cv2.imwrite(str(path), image)
