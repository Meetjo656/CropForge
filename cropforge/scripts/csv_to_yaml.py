import csv
import yaml
import ast
from pathlib import Path

def convert_csv_to_yaml(csv_path, yaml_path):
    print(f"Converting {csv_path} to {yaml_path}...")
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    data = {}
    
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['image_name']
            true_class = row['true_class']
            pred_class = row['pred_class']
            
            if img_name not in data:
                data[img_name] = {
                    'image_name': img_name,
                    'true_class': true_class,
                    'predictions': []
                }
                
            if pred_class != 'None':
                conf = float(row['confidence'])
                bbox_str = row['bbox_xyxy']
                
                # Parse bbox list from string
                try:
                    bbox = ast.literal_eval(bbox_str)
                except:
                    bbox = []
                    
                data[img_name]['predictions'].append({
                    'predicted_class': pred_class,
                    'confidence': conf
                })
                
    yaml_data = {'evaluations': list(data.values())}
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, indent=2)
        
    print(f"Saved {yaml_path}")

if __name__ == "__main__":
    dirs = [
        r"d:\Crop-Forge\runs\detect\random_evaluation",
        r"d:\Crop-Forge\runs\detect\plant_village_50_evaluation"
    ]
    
    for d in dirs:
        d_path = Path(d)
        csv_path = d_path / "predictions.csv"
        yaml_path = d_path / "predictions.yaml"
        convert_csv_to_yaml(csv_path, yaml_path)
