import json
from pathlib import Path

def count_images(root_path):
    stats = {}
    total = 0
    for class_dir in Path(root_path).iterdir():
        if class_dir.is_dir():
            cnt = sum(1 for _ in class_dir.glob('*.jpg'))
            stats[class_dir.name] = cnt
            total += cnt
    return stats, total

def main():
    train_root = r'D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\train'
    test_root = r'D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test'
    train_stats, train_total = count_images(train_root)
    test_stats, test_total = count_images(test_root)
    result = {
        "train": {
            "total_images": train_total,
            "per_class": train_stats
        },
        "test": {
            "total_images": test_total,
            "per_class": test_stats
        }
    }
    # Write JSON summary to file
    out_path = Path(__file__).with_name('plantdoc_analysis.json')
    out_path.write_text(json.dumps(result, indent=2))
    # Print human readable summary
    print('--- Train Set ---')
    print(f'Total images: {train_total}')
    for cls, cnt in sorted(train_stats.items(), key=lambda x: -x[1]):
        print(f'{cls}: {cnt}')
    print('\n--- Test Set ---')
    print(f'Total images: {test_total}')
    for cls, cnt in sorted(test_stats.items(), key=lambda x: -x[1]):
        print(f'{cls}: {cnt}')

if __name__ == '__main__':
    main()
