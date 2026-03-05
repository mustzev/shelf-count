"""
Export YOLOv8m model to ONNX for server-side inference (Proto C).

Usage:
  python3 export_onnx.py                              # Export default .pt → ONNX
  python3 export_onnx.py --model models/custom.pt     # Export specific weights
  python3 export_onnx.py --simplify                    # Simplify ONNX graph (requires onnxslim)
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

TRAINING_DIR = Path(__file__).parent.parent
MODELS_DIR = TRAINING_DIR / "models"

# Default: the pre-trained weights in models/
DEFAULT_PT = MODELS_DIR / "albertferre_shelf-product-identifier_sku-110k_yolov8m.pt"


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8m to ONNX for Proto C server")
    parser.add_argument("--model", type=str, default=str(DEFAULT_PT), help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph (requires onnxslim)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Expected .pt weights file.")
        return

    model = YOLO(str(model_path))

    print(f"Exporting {model_path.name} to ONNX (float32)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
    )
    print(f"ONNX model: {onnx_path}")

    # Copy to training/models/
    onnx_dest = MODELS_DIR / f"{model_path.stem}.onnx"
    if str(Path(onnx_path).resolve()) != str(onnx_dest.resolve()):
        shutil.copy2(onnx_path, onnx_dest)
        print(f"Copied to: {onnx_dest}")


if __name__ == "__main__":
    main()
