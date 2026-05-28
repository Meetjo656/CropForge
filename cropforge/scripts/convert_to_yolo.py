import os, shutil, random, yaml, cv2, numpy as np
from pathlib import Path
from cropforge.data.preprocessing.leaf_preprocessor import LeafPreprocessor

class YOLODatasetConverter:
    def __init__(self, src_dir, dest_dir, split_ratio=(0.7, 0.2, 0.1), seed=42):
        self.src_dir = Path(src_dir)
        self.dest_dir = Path(dest_dir)
        self.split_ratio = split_ratio
        self.seed = seed
        self.class_to_id = {}
        self.id_to_class = {}
        random.seed(self.seed)

    def create_directory_structure(self):
        for split in ['train', 'val', 'test']:
            for sub in ['images', 'labels']:
                dir_path = self.dest_dir / split / sub
                if dir_path.exists():
                    print(f"[WARN] Directory {dir_path} exists. Files may be overwritten.")
                dir_path.mkdir(parents=True, exist_ok=True)

    def load_class_names(self):
        class_dirs = [d for d in self.src_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        return self.class_to_id

    def split_data(self, files_by_class):
        split_files = {'train': [], 'val': [], 'test': []}
        for class_name, files in files_by_class.items():
            random.shuffle(files)
            n = len(files)
            n_train = int(n * self.split_ratio[0])
            n_val = int(n * self.split_ratio[1])
            split_files['train'].extend([(f, class_name) for f in files[:n_train]])
            split_files['val'].extend([(f, class_name) for f in files[n_train:n_train+n_val]])
            split_files['test'].extend([(f, class_name) for f in files[n_train+n_val:]])
        return split_files

    def process_and_save_item(self, img_path, class_id, split_set):
        # Preprocess image
        try:
            pre = LeafPreprocessor(str(img_path), target_size=(640, 640), use_masking=True)
            img = pre.preprocess()['normalized_final']
        except Exception as e:
            print(f"[ERROR] Skipping {img_path}: {e}")
            return
        # Save image
        img_name = img_path.stem + ".jpg"
        img_out_path = self.dest_dir / split_set / 'images' / img_name
        cv2.imwrite(str(img_out_path), img)
        # Save YOLO label
        label_out_path = self.dest_dir / split_set / 'labels' / (img_path.stem + ".txt")
        with open(label_out_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    def generate_dataset_yaml(self):
        yaml_dict = {
            'path': str(self.dest_dir.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.id_to_class
        }
        yaml_path = self.dest_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dict, f)
        print(f"[INFO] Wrote dataset.yaml to {yaml_path}")

    def convert(self):
        self.create_directory_structure()
        self.load_class_names()
        # Gather all image files by class
        files_by_class = {}
        for class_name in self.class_to_id:
            class_dir = self.src_dir / class_name
            files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            files_by_class[class_name] = files
        split_files = self.split_data(files_by_class)
        for split_set, items in split_files.items():
            for img_path, class_name in items:
                class_id = self.class_to_id[class_name]
                self.process_and_save_item(img_path, class_id, split_set)
        self.generate_dataset_yaml()
        print("[INFO] Conversion complete.")

if __name__ == "__main__":
    # 1. Get the absolute directory where this script resides
    # (D:\Crop-Forge\cropforge\scripts)
    SCRIPT_DIR = Path(__file__).resolve().parent

    # 2. Navigate to the cropforge root (D:\Crop-Forge\cropforge)
    # Only go up ONE level since datasets is inside cropforge/
    CROPFORGE_ROOT = SCRIPT_DIR.parent

    # 3. Define absolute paths relative to cropforge root
    src = CROPFORGE_ROOT / "datasets" / "raw" / "PlantVillage"
    dest = CROPFORGE_ROOT / "datasets" / "processed" / "yolo"

    # Sanity check for developers
    print(f"[INFO] Looking for raw data at: {src.resolve()}")
    print(f"[INFO] Outputting YOLO data to: {dest.resolve()}")

    if not src.exists():
        print(f"[CRITICAL ERROR] Source directory does not exist! Please check: {src}")
    else:
        # Run the conversion safely
        converter = YOLODatasetConverter(src, dest)
        converter.convert()