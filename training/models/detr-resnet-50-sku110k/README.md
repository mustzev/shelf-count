---
license: apache-2.0
tags:
- object-detection
- vision
datasets:
- sku110k
widget:
  - src: >-
      https://github.com/Isalia20/DETR-finetune/blob/main/IMG_3507.jpg?raw=true
    example_title: StoreExample(Not from SKU110K Dataset)
---

# DETR (End-to-End Object Detection) model with ResNet-50 backbone trained on SKU110K Dataset with 400 num_queries

DEtection TRansformer (DETR) model trained end-to-end on SKU110K object detection (8k annotated images) dataset. Main difference compared to the original model is it having **400** num_queries and it being pretrained on SKU110K dataset.

### How to use

Here is how to use this model:

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageOps
import requests

url = "https://github.com/Isalia20/DETR-finetune/blob/main/IMG_3507.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image = ImageOps.exif_transpose(image)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("isalia99/detr-resnet-50-sku110k")
model = model.eval()
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.8
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
```
This should output:
```
Detected LABEL_1 with confidence 0.983 at location [665.49, 480.05, 708.15, 650.11]
Detected LABEL_1 with confidence 0.938 at location [204.99, 1405.9, 239.9, 1546.5]
...
Detected LABEL_1 with confidence 0.998 at location [772.85, 169.49, 829.67, 372.18]
Detected LABEL_1 with confidence 0.999 at location [828.28, 1475.16, 874.37, 1593.43]
```

Currently, both the feature extractor and model support PyTorch. 

## Training data

The DETR model was trained on [SKU110K Dataset](https://github.com/eg4000/SKU110K_CVPR19), a dataset consisting of **8,219/588/2,936** annotated images for training/validation/test respectively.

## Training procedure
### Training

The model was trained for 140 epochs on 1 RTX 4060 Ti GPU(Finetuning decoder only) with batch size of 8 and 70 epochs(finetuning the whole network) with batch size of 3 and accumulating gradients for 3 steps.

## Evaluation results

This model achieves an mAP of **58.9** on SKU110k validation set. Result was calculated with torchmetrics MeanAveragePrecision class.

## Training Code

Code is released in this repository [Repo Link](https://github.com/Isalia20/DETR-finetune/tree/main). However it's not finalized/tested well yet but the main stuff is in the code.