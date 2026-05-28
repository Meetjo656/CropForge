import os
import shutil
import random
import yaml
import cv2
import hashlib
import sys
from pathlib import Path

# Add project root to Python path so absolute imports work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from cropforge.data.preprocessing.leaf_preprocessor import LeafPreprocessor

class DatasetSanitizer:
    def __init__(self, src_dir, dest_dir, split_ratio=(0.7, 0.2, 0.1), seed=42):
        self.src_dir = Path(src_dir)
        self.dest_dir = Path(dest_dir)
        self.split_ratio = split_ratio
        self.seed = seed
        self.class_to_id = {}
        self.id_to_class = {}
        random.seed(self.seed)

    def clean_raw_dataset(self):
        print("=== Step 1: Cleaning Raw Dataset ===")
        # Remove accidental nested class directory PlantVillage if it exists
        accidental_dir = self.src_dir / "PlantVillage"
        if accidental_dir.exists() and accidental_dir.is_dir():
            print(f"[ACCIDENTAL CLASS] Found accidental class directory: {accidental_dir}")
            print(f"[ACCIDENTAL CLASS] Removing {accidental_dir} recursively...")
            shutil.rmtree(accidental_dir)
            print("[ACCIDENTAL CLASS] Accidental class directory successfully removed.")
        else:
            print("[ACCIDENTAL CLASS] No accidental class directory 'PlantVillage' found inside source.")

        # Traverse remaining directories to remove duplicates and corrupted files
        class_dirs = [d for d in self.src_dir.iterdir() if d.is_dir()]
        total_removed_duplicates = 0
        total_removed_corrupted = 0
        seen_hashes = set()

        for class_dir in class_dirs:
            print(f"[SCAN] Scanning class folder: {class_dir.name}")
            # Find all image files
            files = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                files.extend(class_dir.glob(ext))
            
            # Sort files for deterministic duplicate matching
            files = sorted(list(set(files)))

            for img_path in files:
                if not img_path.exists():
                    continue
                
                # Check for duplicate files (using file hash)
                try:
                    with open(img_path, 'rb') as f:
                        data = f.read()
                    file_hash = hashlib.md5(data).hexdigest()
                    
                    if file_hash in seen_hashes:
                        print(f"  [DUPLICATE] Deleting duplicate image: {img_path.name}")
                        os.remove(img_path)
                        total_removed_duplicates += 1
                        continue
                    
                    # Verify if image is corrupted (can cv2 read it?)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError("cv2.imread returned None (corrupted)")
                    
                    # If valid, add hash to seen set
                    seen_hashes.add(file_hash)
                    
                except Exception as e:
                    print(f"  [CORRUPTED] Deleting corrupted image: {img_path.name}. Reason: {e}")
                    try:
                        if img_path.exists():
                            os.remove(img_path)
                        total_removed_corrupted += 1
                    except Exception as ex:
                        print(f"    [ERROR] Failed to delete corrupted file {img_path}: {ex}")

        print(f"[CLEANUP SUMMARY]")
        print(f"  Total duplicates removed: {total_removed_duplicates}")
        print(f"  Total corrupted files removed: {total_removed_corrupted}")
        print("Raw dataset sanitization complete.\n")

    def rebuild_class_mappings(self):
        print("=== Step 2: Rebuilding Class Mappings ===")
        class_dirs = sorted([d for d in self.src_dir.iterdir() if d.is_dir()])
        class_names = [d.name for d in class_dirs]
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        print(f"[MAPPINGS] Rebuilt {len(self.class_to_id)} classes:")
        for name, idx in self.class_to_id.items():
            print(f"  {idx}: {name}")
        print("Class mappings rebuilt successfully.\n")
        return self.class_to_id

    def generate_yolo_dataset(self):
        print("=== Step 3: Rebuilding Processed YOLO Dataset ===")
        # Recreate directory structure
        if self.dest_dir.exists():
            print(f"[DEST] Clearing existing destination directory: {self.dest_dir}")
            # Try to safely clear processed splits
            for split in ['train', 'val', 'test']:
                split_dir = self.dest_dir / split
                if split_dir.exists():
                    shutil.rmtree(split_dir)
        
        for split in ['train', 'val', 'test']:
            for sub in ['images', 'labels']:
                dir_path = self.dest_dir / split / sub
                dir_path.mkdir(parents=True, exist_ok=True)

        # Gather clean files by class
        files_by_class = {}
        for class_name in self.class_to_id:
            class_dir = self.src_dir / class_name
            files = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                files.extend(class_dir.glob(ext))
            files_by_class[class_name] = sorted(list(set(files)))

        # Split data
        split_files = {'train': [], 'val': [], 'test': []}
        for class_name, files in files_by_class.items():
            # Seed-shuffled for consistent split
            random.shuffle(files)
            n = len(files)
            n_train = int(n * self.split_ratio[0])
            n_val = int(n * self.split_ratio[1])
            split_files['train'].extend([(f, class_name) for f in files[:n_train]])
            split_files['val'].extend([(f, class_name) for f in files[n_train:n_train+n_val]])
            split_files['test'].extend([(f, class_name) for f in files[n_train+n_val:]])

        # Process and save images and label files
        for split_set, items in split_files.items():
            print(f"[SPLIT] Processing '{split_set}' split ({len(items)} items)...")
            for img_path, class_name in items:
                class_id = self.class_to_id[class_name]
                
                # Preprocess image
                try:
                    pre = LeafPreprocessor(str(img_path), target_size=(640, 640), use_masking=True)
                    img = pre.preprocess()['normalized_final']
                except Exception as e:
                    print(f"  [ERROR] Skipping preprocessing for {img_path.name}: {e}")
                    continue
                
                # Save image
                img_name = img_path.stem + ".jpg"
                img_out_path = self.dest_dir / split_set / 'images' / img_name
                cv2.imwrite(str(img_out_path), img)
                
                # Save YOLO label (whole image bounding box)
                label_out_path = self.dest_dir / split_set / 'labels' / (img_path.stem + ".txt")
                with open(label_out_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        # Write updated dataset.yaml
        yaml_dict = {
            'path': str(self.dest_dir.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.id_to_class
        }
        yaml_path = self.dest_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)
        
        print(f"[INFO] Successfully updated and wrote dataset.yaml to {yaml_path}")
        print("Processed YOLO dataset generation complete.\n")

    def run(self):
        self.clean_raw_dataset()
        self.rebuild_class_mappings()
        self.generate_yolo_dataset()
        print("=== All Steps Completed Successfully! ===")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    CROPFORGE_ROOT = SCRIPT_DIR.parent

    # Source: raw dataset
    src = CROPFORGE_ROOT / "datasets" / "raw" / "PlantVillage"
    # Destination: processed YOLO dataset
    dest = CROPFORGE_ROOT / "datasets" / "processed" / "yolo"

    print(f"[INFO] Raw Source: {src.resolve()}")
    print(f"[INFO] Processed Dest: {dest.resolve()}")

    if not src.exists():
        print(f"[CRITICAL ERROR] Source directory does not exist! Please check: {src}")
    else:
        sanitizer = DatasetSanitizer(src, dest)
        sanitizer.run()
