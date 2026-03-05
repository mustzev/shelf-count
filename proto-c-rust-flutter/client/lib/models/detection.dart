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

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      width: (json['width'] as num).toDouble(),
      height: (json['height'] as num).toDouble(),
    );
  }
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;
  final String? sku;
  final double? skuConfidence;

  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
    this.sku,
    this.skuConfidence,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
      sku: json['sku'] as String?,
      skuConfidence: json['sku_confidence'] != null
          ? (json['sku_confidence'] as num).toDouble()
          : null,
    );
  }
}

class DetectionResult {
  final String model;
  final int count;
  final List<Detection> detections;
  final int inferenceTimeMs;

  const DetectionResult({
    required this.model,
    required this.count,
    required this.detections,
    required this.inferenceTimeMs,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      model: json['model'] as String,
      count: json['count'] as int,
      detections: (json['detections'] as List)
          .map((d) => Detection.fromJson(d as Map<String, dynamic>))
          .toList(),
      inferenceTimeMs: json['inference_time_ms'] as int,
    );
  }
}
