import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/models/detection.dart';

void main() {
  group('BoundingBox', () {
    test('stores normalized coordinates', () {
      const box = BoundingBox(x: 0.1, y: 0.2, width: 0.3, height: 0.4);
      expect(box.x, 0.1);
      expect(box.y, 0.2);
      expect(box.width, 0.3);
      expect(box.height, 0.4);
    });
  });

  group('Detection', () {
    test('stores label, confidence, and bbox', () {
      const det = Detection(
        label: 'bottle',
        confidence: 0.95,
        bbox: BoundingBox(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
      );
      expect(det.label, 'bottle');
      expect(det.confidence, 0.95);
      expect(det.bbox.x, 0.1);
    });
  });

  group('DetectionResult', () {
    test('count matches detections length', () {
      const result = DetectionResult(
        detections: [
          Detection(
            label: 'bottle',
            confidence: 0.9,
            bbox: BoundingBox(x: 0, y: 0, width: 0.1, height: 0.1),
          ),
        ],
        count: 1,
        inferenceTimeMs: 42.5,
      );
      expect(result.count, 1);
      expect(result.detections.length, 1);
      expect(result.inferenceTimeMs, 42.5);
    });
  });
}
