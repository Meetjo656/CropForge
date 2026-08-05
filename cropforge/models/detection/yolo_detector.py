import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        if self.model is None:
            self.model = YOLO(self.model_path)
            
    def predict(self, image):
        """
        Detect bounding boxes in the image.
        Returns:
            boxes: The bounding boxes predicted.
            class_name: The name of the predicted class (for the highest confidence box).
            confidence: The confidence score for the prediction.
        """
        if self.model is None:
            self.load_model()
            
        results = self.model.predict(source=image, verbose=False)[0]
        
        boxes = None
        class_name = None
        confidence = 0.0
        
        if len(results.boxes) > 0:
            boxes = results.boxes
            best_box = results.boxes[results.boxes.conf.argmax()]
            class_name = self.model.names[int(best_box.cls.item())]
            confidence = float(best_box.conf.item())
            
        return boxes, class_name, confidence
