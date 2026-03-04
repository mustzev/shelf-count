import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/services/labels.dart';

void main() {
  test('has 90 entries', () {
    expect(coco90Labels.length, 90);
  });

  test('first label is person', () {
    expect(coco90Labels[0], 'person');
  });

  test('unused slots are null', () {
    expect(coco90Labels[11], isNull); // COCO ID 12
    expect(coco90Labels[25], isNull); // COCO ID 26
  });

  test('bottle is at index 43', () {
    expect(coco90Labels[43], 'bottle');
  });
}
