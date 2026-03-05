"""Export DETR model from safetensors to ONNX format."""

import argparse
from pathlib import Path

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Export DETR to ONNX")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/detr-resnet-50-sku110k",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/detr-resnet-50-sku110k.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    model = DetrForObjectDetection.from_pretrained(model_dir)
    model.eval()

    # DETR expects pixel_values [B, 3, H, W] and pixel_mask [B, H, W]
    # Use the preprocessor's default size: shortest=800, longest=1333
    dummy_pixel_values = torch.randn(1, 3, 800, 800)
    dummy_pixel_mask = torch.ones(1, 800, 800, dtype=torch.int64)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_pixel_values, dummy_pixel_mask),
        str(output_path),
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "pixel_mask": {0: "batch", 1: "height", 2: "width"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done! Exported to {output_path} ({size_mb:.1f} MB)")
    print(f"Output tensors:")
    print(f"  logits:     [batch, {model.config.num_queries}, {model.config.num_labels + 1}]")
    print(f"  pred_boxes: [batch, {model.config.num_queries}, 4]  (cx, cy, w, h normalized 0-1)")


if __name__ == "__main__":
    main()
