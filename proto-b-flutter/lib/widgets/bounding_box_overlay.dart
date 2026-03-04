import 'package:flutter/material.dart';

import '../models/detection.dart';

class BoundingBoxOverlay extends StatelessWidget {
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;
  final double viewWidth;
  final double viewHeight;

  const BoundingBoxOverlay({
    super.key,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.viewWidth,
    required this.viewHeight,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(viewWidth, viewHeight),
      painter: _BoxPainter(
        detections: detections,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;

  _BoxPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final textStyle = TextStyle(
      color: Colors.black,
      fontSize: 10,
      fontWeight: FontWeight.bold,
      background: Paint()..color = Colors.green,
    );

    for (final det in detections) {
      final rect = Rect.fromLTWH(
        det.bbox.x * size.width,
        det.bbox.y * size.height,
        det.bbox.width * size.width,
        det.bbox.height * size.height,
      );

      canvas.drawRect(rect, paint);

      final textSpan = TextSpan(
        text: ' ${det.label} ${(det.confidence * 100).round()}% ',
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      textPainter.paint(canvas, Offset(rect.left, rect.top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) =>
      detections != old.detections;
}
