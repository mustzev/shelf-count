import '../models/detection.dart';

class AuditRecord {
  final String id;
  final int timestamp;
  final String photoPath;
  final DetectionResult result;

  const AuditRecord({
    required this.id,
    required this.timestamp,
    required this.photoPath,
    required this.result,
  });
}

class StorageService {
  final List<AuditRecord> _audits = [];

  List<AuditRecord> get audits => List.unmodifiable(_audits);

  void save(AuditRecord record) {
    _audits.insert(0, record);
  }

  void clear() {
    _audits.clear();
  }
}
