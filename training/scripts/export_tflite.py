"""
Export trained YOLO26n model to TFLite for on-device inference.

Usage:
  python3 export_tflite.py                    # Export best.pt → TFLite
  python3 export_tflite.py --quantize         # Also create int8 quantized version
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

TRAINING_DIR = Path(__file__).parent.parent
MODELS_DIR = TRAINING_DIR / "models"
RUNS_DIR = TRAINING_DIR / "runs"
BEST_PT = RUNS_DIR / "detect" / "sku110k" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26n to TFLite")
    parser.add_argument("--model", type=str, default=str(BEST_PT), help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--quantize", action="store_true", help="Create int8 quantized version")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first with: python3 train.py")
        return

    model = YOLO(str(model_path))

    # Export float32 TFLite
    print(f"Exporting {model_path.name} to TFLite (float32)...")
    tflite_path = model.export(
        format="tflite",
        imgsz=args.imgsz,
    )
    print(f"Float32 TFLite: {tflite_path}")

    # Copy to models/ dir
    tflite_dest = MODELS_DIR / "sku110k-yolo26n.tflite"
    shutil.copy2(tflite_path, tflite_dest)
    print(f"Copied to: {tflite_dest}")

    if args.quantize:
        print(f"\nExporting {model_path.name} to TFLite (int8 quantized)...")
        tflite_int8_path = model.export(
            format="tflite",
            imgsz=args.imgsz,
            int8=True,
        )
        print(f"Int8 TFLite: {tflite_int8_path}")
        int8_dest = MODELS_DIR / "sku110k-yolo26n-int8.tflite"
        shutil.copy2(tflite_int8_path, int8_dest)
        print(f"Copied to: {int8_dest}")


if __name__ == "__main__":
    main()
