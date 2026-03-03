// EfficientDet-Lite0 outputs 90 class slots mapping to COCO category IDs 1–90.
// COCO IDs are NOT contiguous — 10 IDs are unused (12, 26, 29, 30, 45, 66, 68, 69, 71, 83).
// Index = model output class index (0–89). Value = label or null for unused IDs.
export const COCO_90_LABELS: (string | null)[] = [
  'person', // 0 → COCO 1
  'bicycle', // 1 → COCO 2
  'car', // 2 → COCO 3
  'motorcycle', // 3 → COCO 4
  'airplane', // 4 → COCO 5
  'bus', // 5 → COCO 6
  'train', // 6 → COCO 7
  'truck', // 7 → COCO 8
  'boat', // 8 → COCO 9
  'traffic light', // 9 → COCO 10
  'fire hydrant', // 10 → COCO 11
  null, // 11 → COCO 12 (unused)
  'stop sign', // 12 → COCO 13
  'parking meter', // 13 → COCO 14
  'bench', // 14 → COCO 15
  'bird', // 15 → COCO 16
  'cat', // 16 → COCO 17
  'dog', // 17 → COCO 18
  'horse', // 18 → COCO 19
  'sheep', // 19 → COCO 20
  'cow', // 20 → COCO 21
  'elephant', // 21 → COCO 22
  'bear', // 22 → COCO 23
  'zebra', // 23 → COCO 24
  'giraffe', // 24 → COCO 25
  null, // 25 → COCO 26 (unused)
  'backpack', // 26 → COCO 27
  'umbrella', // 27 → COCO 28
  null, // 28 → COCO 29 (unused)
  null, // 29 → COCO 30 (unused)
  'handbag', // 30 → COCO 31
  'tie', // 31 → COCO 32
  'suitcase', // 32 → COCO 33
  'frisbee', // 33 → COCO 34
  'skis', // 34 → COCO 35
  'snowboard', // 35 → COCO 36
  'sports ball', // 36 → COCO 37
  'kite', // 37 → COCO 38
  'baseball bat', // 38 → COCO 39
  'baseball glove', // 39 → COCO 40
  'skateboard', // 40 → COCO 41
  'surfboard', // 41 → COCO 42
  'tennis racket', // 42 → COCO 43
  'bottle', // 43 → COCO 44
  null, // 44 → COCO 45 (unused)
  'wine glass', // 45 → COCO 46
  'cup', // 46 → COCO 47
  'fork', // 47 → COCO 48
  'knife', // 48 → COCO 49
  'spoon', // 49 → COCO 50
  'bowl', // 50 → COCO 51
  'banana', // 51 → COCO 52
  'apple', // 52 → COCO 53
  'sandwich', // 53 → COCO 54
  'orange', // 54 → COCO 55
  'broccoli', // 55 → COCO 56
  'carrot', // 56 → COCO 57
  'hot dog', // 57 → COCO 58
  'pizza', // 58 → COCO 59
  'donut', // 59 → COCO 60
  'cake', // 60 → COCO 61
  'chair', // 61 → COCO 62
  'couch', // 62 → COCO 63
  'potted plant', // 63 → COCO 64
  'bed', // 64 → COCO 65
  null, // 65 → COCO 66 (unused)
  'dining table', // 66 → COCO 67
  null, // 67 → COCO 68 (unused)
  null, // 68 → COCO 69 (unused)
  'toilet', // 69 → COCO 70
  null, // 70 → COCO 71 (unused)
  'tv', // 71 → COCO 72
  'laptop', // 72 → COCO 73
  'mouse', // 73 → COCO 74
  'remote', // 74 → COCO 75
  'keyboard', // 75 → COCO 76
  'cell phone', // 76 → COCO 77
  'microwave', // 77 → COCO 78
  'oven', // 78 → COCO 79
  'toaster', // 79 → COCO 80
  'sink', // 80 → COCO 81
  'refrigerator', // 81 → COCO 82
  null, // 82 → COCO 83 (unused)
  'book', // 83 → COCO 84
  'clock', // 84 → COCO 85
  'vase', // 85 → COCO 86
  'scissors', // 86 → COCO 87
  'teddy bear', // 87 → COCO 88
  'hair drier', // 88 → COCO 89
  'toothbrush', // 89 → COCO 90
]
