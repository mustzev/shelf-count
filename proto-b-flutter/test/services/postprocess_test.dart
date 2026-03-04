import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/services/postprocess.dart';

void main() {
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
    null,                                           // 42
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
}
