import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/models/detection.dart';
import 'package:proto_b_flutter/services/storage_service.dart';

void main() {
  late StorageService storage;

  setUp(() {
    storage = StorageService();
  });

  test('starts empty', () {
    expect(storage.audits, isEmpty);
  });

  test('saves and retrieves an audit', () {
    final record = AuditRecord(
      id: '1',
      timestamp: DateTime.now().millisecondsSinceEpoch,
      photoPath: '/tmp/photo.jpg',
      result: const DetectionResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 50,
      ),
    );

    storage.save(record);
    expect(storage.audits.length, 1);
    expect(storage.audits.first.id, '1');
  });

  test('most recent audit is first', () {
    storage.save(AuditRecord(
      id: '1',
      timestamp: 100,
      photoPath: '/a.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));
    storage.save(AuditRecord(
      id: '2',
      timestamp: 200,
      photoPath: '/b.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));

    expect(storage.audits.first.id, '2');
  });

  test('clear removes all audits', () {
    storage.save(AuditRecord(
      id: '1',
      timestamp: 100,
      photoPath: '/a.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));
    storage.clear();
    expect(storage.audits, isEmpty);
  });
}
