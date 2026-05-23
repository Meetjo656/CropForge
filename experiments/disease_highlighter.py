import cv2
import sys
import os

# Ensure the src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.image_loader import load_image
from src.preprocessing.masking import hsv_masking, apply_mask
from src.preprocessing.morphology import clean_mask
from src.preprocessing.contours import detect_contours, draw_contours
from src.preprocessing.edges import detect_edges
from src.severity.severity_calculator import calculate_severity

def main():
    # Define image path
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/0b17e71c-fee6-4b5e-9a7d-8ba44d29215f___Com.G_SpM_FL 8549.JPG")
    
    # 1. Load Image
    image = load_image(IMAGE_PATH)
    
    # 2. Masking
    mask = hsv_masking(image)
    
    # 3. Morphology (Clean the mask)
    cleaned_mask = clean_mask(mask)
    
    # Apply the cleaned mask to the original image
    masked_output = apply_mask(image, cleaned_mask)
    
    # 4. Contours
    contours = detect_contours(cleaned_mask)
    contour_image = draw_contours(image, contours)
    
    # 5. Edges
    edges = detect_edges(image)
    
    # 6. Severity Scoring
    severity = calculate_severity(cleaned_mask)
    print(f"Disease Severity: {severity:.2f}%")
    
    # 7. Visualization
    cv2.imshow("Original", image)
    cv2.imshow("Cleaned Mask", cleaned_mask)
    cv2.imshow("Disease Regions", masked_output)
    cv2.imshow("Contours", contour_image)
    cv2.imshow("Edges", edges)
    
    print("Press any key on the image windows to close them and save outputs.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 8. Output Saving
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "cleaned_mask.jpg"), cleaned_mask)
    cv2.imwrite(os.path.join(output_dir, "disease_regions.jpg"), masked_output)
    cv2.imwrite(os.path.join(output_dir, "contours.jpg"), contour_image)
    cv2.imwrite(os.path.join(output_dir, "edges.jpg"), edges)
    print(f"Outputs saved successfully in {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
