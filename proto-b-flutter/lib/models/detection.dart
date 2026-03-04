class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;

  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;

  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });
}

class DetectionResult {
  final List<Detection> detections;
  final int count;
  final double inferenceTimeMs;

  const DetectionResult({
    required this.detections,
    required this.count,
    required this.inferenceTimeMs,
  });
}
