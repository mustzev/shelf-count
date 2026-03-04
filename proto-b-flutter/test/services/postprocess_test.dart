import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/services/postprocess.dart';

void main() {
  // ---------------------------------------------------------------------------
  // processPostNmsOutputs — EfficientDet-Lite0 tests
  // ---------------------------------------------------------------------------

  group('processPostNmsOutputs', () {
    final boxes = List<double>.filled(25 * 4, 0.0);
    final classes = List<double>.filled(25, 0.0);
    final scores = List<double>.filled(25, 0.0);
    const count = 2;

    final labels = <String?>[
      'person', // 0
      'bicycle', // 1
      null, null, null, null, null, null, null, null, // 2-9
      null, null, null, null, null, null, null, null, // 10-17
      null, null, null, null, null, null, null, null, // 18-25
      null, null, null, null, null, null, null, null, // 26-33
      null, null, null, null, null, null, null, null, // 34-41
      null, // 42
      'bottle', // 43
    ];

    setUp(() {
      // Reset
      for (var i = 0; i < boxes.length; i++) boxes[i] = 0;
      for (var i = 0; i < classes.length; i++) classes[i] = 0;
      for (var i = 0; i < scores.length; i++) scores[i] = 0;

      // Detection 0: person at [0.1, 0.2, 0.5, 0.6] conf 0.9
      boxes[0] = 0.1; boxes[1] = 0.2; boxes[2] = 0.5; boxes[3] = 0.6;
      classes[0] = 0.0;
      scores[0] = 0.9;

      // Detection 1: bottle at [0.3, 0.4, 0.7, 0.8] conf 0.7
      boxes[4] = 0.3; boxes[5] = 0.4; boxes[6] = 0.7; boxes[7] = 0.8;
      classes[1] = 43.0;
      scores[1] = 0.7;
    });

    test('parses two detections correctly', () {
      final detections = processPostNmsOutputs(
        boxes: boxes,
        classes: classes,
        scores: scores,
        count: count,
        labels: labels,
        scoreThreshold: 0.4,
      );

      expect(detections.length, 2);

      expect(detections[0].label, 'person');
      expect(detections[0].confidence, 0.9);
      expect(detections[0].bbox.x, closeTo(0.2, 0.001));
      expect(detections[0].bbox.y, closeTo(0.1, 0.001));
      expect(detections[0].bbox.width, closeTo(0.4, 0.001));
      expect(detections[0].bbox.height, closeTo(0.4, 0.001));

      expect(detections[1].label, 'bottle');
      expect(detections[1].confidence, 0.7);
    });

    test('filters by confidence threshold', () {
      final detections = processPostNmsOutputs(
        boxes: boxes,
        classes: classes,
        scores: scores,
        count: count,
        labels: labels,
        scoreThreshold: 0.8,
      );

      expect(detections.length, 1);
      expect(detections[0].label, 'person');
    });

    test('skips null labels', () {
      classes[0] = 2.0; // index 2 is null in our short labels list
      scores[0] = 0.95;

      final detections = processPostNmsOutputs(
        boxes: boxes,
        classes: classes,
        scores: scores,
        count: count,
        labels: labels,
        scoreThreshold: 0.4,
      );

      expect(detections.length, 1);
      expect(detections[0].label, 'bottle');
    });

    test('returns empty list when count is 0', () {
      final detections = processPostNmsOutputs(
        boxes: boxes,
        classes: classes,
        scores: scores,
        count: 0,
        labels: labels,
        scoreThreshold: 0.4,
      );

      expect(detections, isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // processYoloOutputs — YOLOv8 single-class tests
  // ---------------------------------------------------------------------------

  group('processYoloOutputs', () {
    // Build a synthetic [1, 5, 8400] row-major output buffer.
    // Layout: row r, col i  ->  index = r * 8400 + i
    //   row 0: cx (normalized 0–1)
    //   row 1: cy
    //   row 2: w
    //   row 3: h
    //   row 4: confidence
    const numBoxes = 8400;
    const numValues = 5;

    List<double> makeRawOutput() =>
        List<double>.filled(numValues * numBoxes, 0.0);

    void writeBox(
      List<double> buf,
      int index, {
      required double cx,
      required double cy,
      required double w,
      required double h,
      required double score,
    }) {
      buf[0 * numBoxes + index] = cx;
      buf[1 * numBoxes + index] = cy;
      buf[2 * numBoxes + index] = w;
      buf[3 * numBoxes + index] = h;
      buf[4 * numBoxes + index] = score;
    }

    test('parses known boxes and returns correct label and coordinates', () {
      final raw = makeRawOutput();

      // Box 0: centred at (0.5, 0.5), 0.1x0.1, conf 0.8
      writeBox(raw, 0, cx: 0.5, cy: 0.5, w: 0.1, h: 0.1, score: 0.8);

      // Box 1: centred at (0.2, 0.1), 0.05x0.05, conf 0.6
      writeBox(raw, 1, cx: 0.2, cy: 0.1, w: 0.05, h: 0.05, score: 0.6);

      final detections = processYoloOutputs(
        rawOutput: raw,
        numBoxes: numBoxes,
        numValues: numValues,
        label: 'object',
        scoreThreshold: 0.25,
        iouThreshold: 0.5,
      );

      expect(detections.length, 2);

      // Results are sorted by confidence descending.
      expect(detections[0].label, 'object');
      expect(detections[0].confidence, closeTo(0.8, 0.001));

      // Box 0: cx=0.5, cy=0.5, w=0.1, h=0.1
      //   x = 0.5 - 0.05 = 0.45,  y = 0.5 - 0.05 = 0.45
      expect(detections[0].bbox.x, closeTo(0.45, 0.001));
      expect(detections[0].bbox.y, closeTo(0.45, 0.001));
      expect(detections[0].bbox.width, closeTo(0.1, 0.001));
      expect(detections[0].bbox.height, closeTo(0.1, 0.001));

      expect(detections[1].label, 'object');
      expect(detections[1].confidence, closeTo(0.6, 0.001));

      // Box 1: cx=0.2, cy=0.1, w=0.05, h=0.05
      //   x = 0.2 - 0.025 = 0.175,  y = 0.1 - 0.025 = 0.075
      expect(detections[1].bbox.x, closeTo(0.175, 0.001));
      expect(detections[1].bbox.y, closeTo(0.075, 0.001));
      expect(detections[1].bbox.width, closeTo(0.05, 0.001));
      expect(detections[1].bbox.height, closeTo(0.05, 0.001));
    });

    test('filters boxes below confidence threshold', () {
      final raw = makeRawOutput();

      writeBox(raw, 0, cx: 0.5, cy: 0.5, w: 0.1, h: 0.1, score: 0.8);
      writeBox(raw, 1, cx: 0.2, cy: 0.2, w: 0.06, h: 0.06, score: 0.1);

      final detections = processYoloOutputs(
        rawOutput: raw,
        numBoxes: numBoxes,
        numValues: numValues,
        label: 'object',
        scoreThreshold: 0.25,
        iouThreshold: 0.5,
      );

      expect(detections.length, 1);
      expect(detections[0].confidence, closeTo(0.8, 0.001));
    });

    test('NMS suppresses lower-confidence box when boxes overlap heavily', () {
      final raw = makeRawOutput();

      writeBox(raw, 0, cx: 0.5, cy: 0.5, w: 0.3, h: 0.3, score: 0.9);
      writeBox(raw, 1, cx: 0.51, cy: 0.51, w: 0.3, h: 0.3, score: 0.7);

      final detections = processYoloOutputs(
        rawOutput: raw,
        numBoxes: numBoxes,
        numValues: numValues,
        label: 'object',
        scoreThreshold: 0.25,
        iouThreshold: 0.5,
      );

      expect(detections.length, 1);
      expect(detections[0].confidence, closeTo(0.9, 0.001));
    });

    test('NMS keeps both boxes when they do not overlap', () {
      final raw = makeRawOutput();

      writeBox(raw, 0, cx: 0.1, cy: 0.1, w: 0.1, h: 0.1, score: 0.85);
      writeBox(raw, 1, cx: 0.9, cy: 0.9, w: 0.1, h: 0.1, score: 0.75);

      final detections = processYoloOutputs(
        rawOutput: raw,
        numBoxes: numBoxes,
        numValues: numValues,
        label: 'object',
        scoreThreshold: 0.25,
        iouThreshold: 0.5,
      );

      expect(detections.length, 2);
    });

    test('returns empty list when all boxes are below threshold', () {
      final raw = makeRawOutput();

      final detections = processYoloOutputs(
        rawOutput: raw,
        numBoxes: numBoxes,
        numValues: numValues,
        label: 'object',
        scoreThreshold: 0.25,
        iouThreshold: 0.5,
      );

      expect(detections, isEmpty);
    });
  });
}
