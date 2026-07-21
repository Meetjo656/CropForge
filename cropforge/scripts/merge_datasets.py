import os
import shutil
import random
import yaml
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from cropforge.data.preprocessing.leaf_preprocessor import LeafPreprocessor

UNIFIED_CLASSES = sorted([
    'Apple_Scab', 'Apple_healthy', 'Apple_rust', 'Blueberry_healthy', 'Cherry_healthy', 
    'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust', 'Grape_black_rot', 'Grape_healthy', 
    'Peach_healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry_healthy', 
    'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
])

CLASS_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

PV_MAPPING = {c: c for c in UNIFIED_CLASSES if c.startswith('Pepper') or c.startswith('Potato') or c.startswith('Tomato')}

PD_MAPPING = {
    "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper__bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato leaf": "Tomato_healthy",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Apple Scab Leaf": "Apple_Scab",
    "Apple leaf": "Apple_healthy",
    "Apple rust leaf": "Apple_rust",
    "Blueberry leaf": "Blueberry_healthy",
    "Cherry leaf": "Cherry_healthy",
    "Corn Gray leaf spot": "Corn_Gray_leaf_spot",
    "Corn leaf blight": "Corn_leaf_blight",
    "Corn rust leaf": "Corn_rust",
    "Peach leaf": "Peach_healthy",
    "Raspberry leaf": "Raspberry_healthy",
    "Soyabean leaf": "Soybean_healthy",
    "Squash Powdery mildew leaf": "Squash_Powdery_mildew",
    "Strawberry leaf": "Strawberry_healthy",
    "grape leaf": "Grape_healthy",
    "grape leaf black rot": "Grape_black_rot"
}

class MergedDatasetConverter:
    def __init__(self, dest_dir, seed=42):
        self.dest_dir = Path(dest_dir)
        self.seed = seed
        random.seed(self.seed)

    def create_directory_structure(self):
        for split in ['train', 'val', 'test']:
            for sub in ['images', 'labels']:
                dir_path = self.dest_dir / split / sub
                if dir_path.exists():
                    print(f"[WARN] Directory {dir_path} exists. Files may be overwritten.")
                dir_path.mkdir(parents=True, exist_ok=True)

    def process_and_save_item(self, img_path, class_id, split_set):
        try:
            # Bypass heavy segmentation and use lightning-fast OpenCV resize
            img = cv2.imread(str(img_path))
            if img is None:
                return
            img = cv2.resize(img, (640, 640))
        except Exception as e:
            return
            
        img_name = f"{img_path.stem}_{random.randint(1000, 9999)}.jpg"
        img_out_path = self.dest_dir / split_set / 'images' / img_name
        cv2.imwrite(str(img_out_path), img)
        
        label_out_path = self.dest_dir / split_set / 'labels' / (img_out_path.stem + ".txt")
        with open(label_out_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    def convert_dataset(self, src_dir, mapping_dict, is_already_split=False):
        src_dir = Path(src_dir)
        
        if is_already_split:
            for split in ['train', 'test']:
                split_dir = src_dir / split
                if not split_dir.exists():
                    continue
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir() or class_dir.name not in mapping_dict:
                        continue
                        
                    class_id = CLASS_TO_ID[mapping_dict[class_dir.name]]
                    files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                    
                    for f in files:
                        self.process_and_save_item(f, class_id, split)
        else:
            files_by_class = {}
            for class_dir in src_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name not in mapping_dict:
                    continue
                files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                files_by_class[mapping_dict[class_dir.name]] = files
                
            for class_name, files in files_by_class.items():
                random.shuffle(files)
                n = len(files)
                n_train = int(n * 0.7)
                n_val = int(n * 0.2)
                
                class_id = CLASS_TO_ID[class_name]
                for f in files[:n_train]:
                    self.process_and_save_item(f, class_id, 'train')
                for f in files[n_train:n_train+n_val]:
                    self.process_and_save_item(f, class_id, 'val')
                for f in files[n_train+n_val:]:
                    self.process_and_save_item(f, class_id, 'test')

    def generate_dataset_yaml(self):
        yaml_dict = {
            'path': str(self.dest_dir.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': ID_TO_CLASS
        }
        yaml_path = self.dest_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dict, f)
        print(f"[INFO] Wrote dataset.yaml to {yaml_path}")
    

if __name__ == "__main__":
    import shutil
    
    CROPFORGE_ROOT = Path(r"D:\Crop-Forge\cropforge")
    dest = CROPFORGE_ROOT / "datasets" / "processed" / "merged_balanced"
    
    if dest.exists():
        shutil.rmtree(dest)
        
    converter = MergedDatasetConverter(dest)
    converter.create_directory_structure()
    
    print("[INFO] Processing PlantVillage...")
    pv_src = CROPFORGE_ROOT / "datasets" / "raw" / "PlantVillage"
    converter.convert_dataset(pv_src, PV_MAPPING, is_already_split=False)
    
    print("[INFO] Processing PlantDoc...")
    pd_src = CROPFORGE_ROOT / "datasets" / "raw" / "PlantDoc"
    converter.convert_dataset(pd_src, PD_MAPPING, is_already_split=True)
    
    converter.generate_dataset_yaml()
    print("[INFO] Done merging datasets.")
