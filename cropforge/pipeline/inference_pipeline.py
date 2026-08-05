import cv2
from pathlib import Path
from cropforge.models.detection.yolo_detector import YOLODetector
from cropforge.models.segmentation.sam2_segmenter import SAM2Segmenter
from cropforge.pipeline.severity import calculate_severity

class SegmentPipeline:
    def __init__(self, detector, segmenter):
        self.detector = detector
        self.segmenter = segmenter

    def predict(self, image):
        """
        Run the full segmentation pipeline.
        Returns the detections and the segmentation masks.
        """
        boxes, class_name, confidence = self.detector.predict(image)
        
        masks = None
        if boxes is not None and len(boxes) > 0:
            masks = self.segmenter.predict(image, boxes)
            
        return boxes, class_name, confidence, masks

def process_image(image_path, output_dir="outputs/pipeline"):
    """
    Entry point for the full disease detection pipeline.
    """
    # 1. Initialize Pipeline
    detector = YOLODetector(model_path=r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt")
    segmenter = SAM2Segmenter(model_path="sam2.1_t.pt")
    pipeline = SegmentPipeline(detector, segmenter)
    
    # 2. Read Image
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # 3. Detect and Segment
    boxes, class_name, confidence, masks = pipeline.predict(image)
    
    if class_name is None:
        return {
            "disease": None,
            "confidence": 0.0,
            "severity": 0.0,
            "error": "No disease detected"
        }
        
    # 4. Calculate Severity
    severity_data = calculate_severity(masks)
    
    # 5. Save Outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    overlay_img = segmenter.overlay(image, masks)
    
    base_name = image_path.stem
    mask_path = out_dir / f"{base_name}_mask.jpg"
    overlay_path = out_dir / f"{base_name}_overlay.jpg"
    
    # Assuming mask is visualizable. The SAM2Segmenter could be updated to save just the mask.
    # For now, we'll save the overlay.
    segmenter.save(overlay_img, overlay_path)
    # We will also save a pure mask if needed, but the overlay provides a good visual.
    
    # 6. Return JSON
    return {
        "disease": class_name,
        "confidence": round(confidence, 4),
        "severity": severity_data["severity"],
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path)
    }

if __name__ == "__main__":
    # Test the pipeline
    test_image = r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test\Tomato leaf late blight\0a3d6021-995a-4934-9ca1-19d6d8dbdeeb___RS_Late.B 6245.JPG"
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    try:
        result = process_image(test_image)
        print("Pipeline Result:")
        import json
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Pipeline failed: {e}")
