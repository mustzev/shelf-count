# Fine-Tuning Detection Models for Custom Shelves

## When You Need Fine-Tuning

Both YOLOv8 and DETR models trained on SKU-110K are **class-agnostic** — they detect "a product on a shelf" regardless of brand, label language, or packaging. They should generalize well to most supermarket environments, including non-Western markets.

Fine-tuning becomes useful when:
- Very small items (spice packets, candy bars) packed extremely tight
- Non-standard displays (hanging products, irregular arrangements)
- Glass refrigerator doors causing reflections
- Shelf density or product shapes differ significantly from SKU-110K

## What Doesn't Affect Detection

| Factor | Why |
|--------|-----|
| Label language (Cyrillic, Mongolian, etc.) | Models detect edges/shapes, not text |
| Local/unfamiliar brands | Class-agnostic — doesn't matter what the product is |
| Standard product shapes (box, bottle, can) | Well-represented in SKU-110K |

## What Can Affect Detection

| Factor | Risk |
|--------|------|
| Shelf density very different from training data | Medium |
| Unusual packaging (bags, pouches, odd shapes) | Medium |
| Lighting / reflections | Medium |
| Very small or very large items | Medium |

## Fine-Tuning Process

1. **Collect** — photograph target shelves (50–200 images)
2. **Annotate** — draw bounding boxes around every product (tools: [Roboflow](https://roboflow.com), [CVAT](https://cvat.ai), Label Studio)
3. **Fine-tune** — resume training from existing SKU-110K weights

Annotation is the bottleneck (~2–5 min per dense shelf image). Training takes 1–2 hours on a single GPU.

### YOLOv8

```python
from ultralytics import YOLO

model = YOLO("yolov8m-sku110k.pt")  # start from SKU-110K weights
model.train(data="your_dataset.yaml", epochs=20)
```

### DETR

Use HuggingFace Trainer starting from the `isalia99/detr-resnet-50-sku110k` checkpoint. See HuggingFace DETR fine-tuning tutorial for details.

## YOLO vs DETR for Domain Shift

- **YOLOv8m** — more robust to domain shift for dense shelf scenes; designed for this kind of task
- **DETR** — more sensitive to object density patterns; `num_queries=400` handles dense shelves but may struggle more if layout differs significantly from training data
