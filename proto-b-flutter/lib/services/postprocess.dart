import 'dart:math' as math;

import '../models/detection.dart';

/// Parse post-NMS EfficientDet-Lite0 outputs into Detection objects.
///
/// EfficientDet-Lite0 output tensors:
///   [0] boxes   — [1, maxDet, 4] as [y1, x1, y2, x2] normalized 0–1
///   [1] classes  — [1, maxDet] class index as float
///   [2] scores   — [1, maxDet] confidence 0–1
///   [3] count    — [1] number of valid detections
List<Detection> processPostNmsOutputs({
  required List<double> boxes,
  required List<double> classes,
  required List<double> scores,
  required int count,
  required List<String?> labels,
  required double scoreThreshold,
}) {
  final detections = <Detection>[];

  for (var i = 0; i < count; i++) {
    if (scores[i] < scoreThreshold) continue;

    final classIdx = classes[i].round();
    if (classIdx < 0 || classIdx >= labels.length) continue;
    final label = labels[classIdx];
    if (label == null) continue;

    final y1 = boxes[i * 4];
    final x1 = boxes[i * 4 + 1];
    final y2 = boxes[i * 4 + 2];
    final x2 = boxes[i * 4 + 3];

    detections.add(Detection(
      label: label,
      confidence: scores[i],
      bbox: BoundingBox(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
    ));
  }

  return detections;
}

/// Parse YOLOv8 TFLite output into Detection objects.
///
/// YOLOv8 single-class output tensor: [1, 5, 8400]
///   — 8400 candidate boxes, each column: [cx, cy, w, h, confidence]
///   — coordinates are already normalized 0–1
///
/// Steps: transpose → confidence filter → NMS → clamp to 0–1.
List<Detection> processYoloOutputs({
  required List<double> rawOutput,
  required int numBoxes,
  required int numValues,
  required String label,
  required double scoreThreshold,
  double iouThreshold = 0.5,
}) {
  // rawOutput is [1, numValues, numBoxes] flattened row-major.
  // Row 0: all cx values, Row 1: all cy values, etc.
  // Transpose to per-box access: box[i] = [cx, cy, w, h, score]
  final candidates = <_Box>[];

  for (var i = 0; i < numBoxes; i++) {
    final score = rawOutput[4 * numBoxes + i]; // row 4, col i
    if (score < scoreThreshold) continue;

    final cx = rawOutput[0 * numBoxes + i];
    final cy = rawOutput[1 * numBoxes + i];
    final w = rawOutput[2 * numBoxes + i];
    final h = rawOutput[3 * numBoxes + i];

    candidates.add(_Box(
      x: cx - w / 2,
      y: cy - h / 2,
      width: w,
      height: h,
      score: score,
    ));
  }

  // Sort by confidence descending
  candidates.sort((a, b) => b.score.compareTo(a.score));

  // Non-Maximum Suppression
  final kept = <_Box>[];
  final suppressed = List<bool>.filled(candidates.length, false);

  for (var i = 0; i < candidates.length; i++) {
    if (suppressed[i]) continue;
    kept.add(candidates[i]);

    for (var j = i + 1; j < candidates.length; j++) {
      if (suppressed[j]) continue;
      if (_iou(candidates[i], candidates[j]) > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return kept
      .map((b) => Detection(
            label: label,
            confidence: b.score,
            bbox: BoundingBox(
              x: b.x.clamp(0.0, 1.0),
              y: b.y.clamp(0.0, 1.0),
              width: b.width.clamp(0.0, 1.0),
              height: b.height.clamp(0.0, 1.0),
            ),
          ))
      .toList();
}

class _Box {
  final double x, y, width, height, score;
  const _Box({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.score,
  });
}

double _iou(_Box a, _Box b) {
  final x1 = math.max(a.x, b.x);
  final y1 = math.max(a.y, b.y);
  final x2 = math.min(a.x + a.width, b.x + b.width);
  final y2 = math.min(a.y + a.height, b.y + b.height);

  if (x2 <= x1 || y2 <= y1) return 0.0;

  final intersection = (x2 - x1) * (y2 - y1);
  final areaA = a.width * a.height;
  final areaB = b.width * b.height;
  final union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0.0;
}
