/// EfficientDet-Lite0 outputs 90 class slots mapping to COCO category IDs 1–90.
/// Index = model output class index (0–89). Value = label or null for unused IDs.
const List<String?> coco90Labels = [
  'person',        // 0
  'bicycle',       // 1
  'car',           // 2
  'motorcycle',    // 3
  'airplane',      // 4
  'bus',           // 5
  'train',         // 6
  'truck',         // 7
  'boat',          // 8
  'traffic light',  // 9
  'fire hydrant',  // 10
  null,            // 11 (COCO 12)
  'stop sign',     // 12
  'parking meter', // 13
  'bench',         // 14
  'bird',          // 15
  'cat',           // 16
  'dog',           // 17
  'horse',         // 18
  'sheep',         // 19
  'cow',           // 20
  'elephant',      // 21
  'bear',          // 22
  'zebra',         // 23
  'giraffe',       // 24
  null,            // 25 (COCO 26)
  'backpack',      // 26
  'umbrella',      // 27
  null,            // 28 (COCO 29)
  null,            // 29 (COCO 30)
  'handbag',       // 30
  'tie',           // 31
  'suitcase',      // 32
  'frisbee',       // 33
  'skis',          // 34
  'snowboard',     // 35
  'sports ball',   // 36
  'kite',          // 37
  'baseball bat',  // 38
  'baseball glove', // 39
  'skateboard',    // 40
  'surfboard',     // 41
  'tennis racket', // 42
  'bottle',        // 43
  null,            // 44 (COCO 45)
  'wine glass',    // 45
  'cup',           // 46
  'fork',          // 47
  'knife',         // 48
  'spoon',         // 49
  'bowl',          // 50
  'banana',        // 51
  'apple',         // 52
  'sandwich',      // 53
  'orange',        // 54
  'broccoli',      // 55
  'carrot',        // 56
  'hot dog',       // 57
  'pizza',         // 58
  'donut',         // 59
  'cake',          // 60
  'chair',         // 61
  'couch',         // 62
  'potted plant',  // 63
  'bed',           // 64
  null,            // 65 (COCO 66)
  'dining table',  // 66
  null,            // 67 (COCO 68)
  null,            // 68 (COCO 69)
  'toilet',        // 69
  null,            // 70 (COCO 71)
  'tv',            // 71
  'laptop',        // 72
  'mouse',         // 73
  'remote',        // 74
  'keyboard',      // 75
  'cell phone',    // 76
  'microwave',     // 77
  'oven',          // 78
  'toaster',       // 79
  'sink',          // 80
  'refrigerator',  // 81
  null,            // 82 (COCO 83)
  'book',          // 83
  'clock',         // 84
  'vase',          // 85
  'scissors',      // 86
  'teddy bear',    // 87
  'hair drier',    // 88
  'toothbrush',    // 89
];
