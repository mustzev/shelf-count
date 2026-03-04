"""
Train YOLO26n on SKU-110K for retail product detection.

YOLO26n: Ultralytics' latest (Jan 2026) — NMS-free, 43% faster on edge,
better small-object detection via STAL.

Usage:
  python3 train.py              # Train on SKU-110K
  python3 train.py --epochs 50  # Custom epoch count
  python3 train.py --resume     # Resume interrupted training
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent / "datasets" / "sku110k" / "SKU 110k.v6-original_raw-images.yolo26"
MODELS_DIR = Path(__file__).parent / "models"
RUNS_DIR = Path(__file__).parent / "runs"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n on SKU-110K")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("Dataset not prepared yet. Run prepare_sku110k.py first.")
        return

    if args.resume:
        last = RUNS_DIR / "detect" / "sku110k" / "weights" / "last.pt"
        if not last.exists():
            print(f"No checkpoint found at {last}")
            return
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        # YOLO26n: NMS-free, small-object-aware, optimized for edge
        model = YOLO(str(MODELS_DIR / "yolo26n.pt"))
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device="mps",  # Apple Metal GPU
            project=str(RUNS_DIR / "detect"),
            name="sku110k",
            patience=15,       # Early stopping
            save_period=10,    # Save checkpoint every 10 epochs
            plots=True,
            exist_ok=True,
        )

    print("\nTraining complete!")
    print(f"Best weights: {RUNS_DIR / 'detect' / 'sku110k' / 'weights' / 'best.pt'}")
    print("Next step: python3 export_tflite.py")


if __name__ == "__main__":
    main()
