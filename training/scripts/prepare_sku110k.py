"""
Download and prepare SKU-110K dataset in YOLO format.

SKU-110K: ~11K images of densely packed retail shelves.
Single class: "product" — detects every item regardless of type.

Original paper: https://github.com/eg4000/SKU110K_CVPR19
"""

import csv
import os
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

TRAINING_DIR = Path(__file__).parent.parent
DATASET_DIR = TRAINING_DIR / "datasets" / "sku110k"

# SKU-110K annotations are CSV with columns:
# image_name, x1, y1, x2, y2, class, image_width, image_height
ANNOTATIONS_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"

# Images hosted on Google Drive (large ~30GB).
# For practical training, we'll use a Roboflow-hosted subset or
# download the full set depending on user preference.

def convert_annotations(csv_path: Path, images_dir: Path, output_dir: Path, split: str):
    """Convert SKU-110K CSV annotations to YOLO format."""
    labels_dir = output_dir / split / "labels"
    imgs_out = output_dir / split / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs_out.mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    annotations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name = row[0]
            try:
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                img_w, img_h = float(row[6]), float(row[7])
            except (ValueError, IndexError):
                continue

            # Convert to YOLO format: class x_center y_center width height (normalized)
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            annotations[img_name].append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    count = 0
    for img_name, labels in annotations.items():
        img_src = images_dir / img_name
        if not img_src.exists():
            continue

        # Symlink image
        img_dst = imgs_out / img_name
        if not img_dst.exists():
            os.symlink(img_src.resolve(), img_dst)

        # Write YOLO label file
        label_name = Path(img_name).stem + ".txt"
        label_path = labels_dir / label_name
        label_path.write_text("\n".join(labels) + "\n")
        count += 1

    print(f"  {split}: {count} images, {sum(len(v) for v in annotations.values())} annotations")
    return count


def main():
    print("SKU-110K Dataset Preparation")
    print("=" * 40)

    # Check if already prepared
    yaml_path = DATASET_DIR / "sku110k.yaml"
    if yaml_path.exists():
        print(f"Dataset already prepared at {DATASET_DIR}")
        print(f"Config: {yaml_path}")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Check for manually downloaded files
    images_dir = DATASET_DIR / "images"
    annot_dir = DATASET_DIR / "annotations"

    if not (annot_dir / "annotations_train.csv").exists():
        print("\nSKU-110K requires manual download due to hosting restrictions.")
        print("\nStep 1: Download annotations")
        print(f"  URL: {ANNOTATIONS_URL}")
        print(f"  Extract to: {annot_dir}/")
        print("  Expected files: annotations_train.csv, annotations_val.csv, annotations_test.csv")
        print()
        print("Step 2: Download images")
        print("  From: https://github.com/eg4000/SKU110K_CVPR19")
        print("  Follow the Google Drive links in the README")
        print(f"  Extract to: {images_dir}/")
        print()
        print("Step 3: Re-run this script")
        print(f"  python3 {__file__}")
        return

    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"Images directory not found or empty: {images_dir}")
        print("Download images from: https://github.com/eg4000/SKU110K_CVPR19")
        return

    # Convert to YOLO format
    print("\nConverting annotations to YOLO format...")
    yolo_dir = DATASET_DIR / "yolo"

    for split, csv_name in [("train", "annotations_train.csv"),
                             ("val", "annotations_val.csv"),
                             ("test", "annotations_test.csv")]:
        csv_path = annot_dir / csv_name
        if csv_path.exists():
            convert_annotations(csv_path, images_dir, yolo_dir, split)
        else:
            print(f"  {split}: skipped (no {csv_name})")

    # Write YOLO dataset config
    yaml_content = f"""# SKU-110K dataset for retail product detection
path: {yolo_dir.resolve()}
train: train/images
val: val/images
test: test/images

names:
  0: product
"""
    yaml_path.write_text(yaml_content)
    print(f"\nDataset config written to: {yaml_path}")
    print("Ready for training!")


if __name__ == "__main__":
    main()
