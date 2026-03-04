import 'dart:io';

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../widgets/bounding_box_overlay.dart';

class _ImageWithOverlay extends StatefulWidget {
  final String photoPath;
  final List<Detection> detections;

  const _ImageWithOverlay({
    required this.photoPath,
    required this.detections,
  });

  @override
  State<_ImageWithOverlay> createState() => _ImageWithOverlayState();
}

class _ImageWithOverlayState extends State<_ImageWithOverlay> {
  Size? _imageSize;

  @override
  void initState() {
    super.initState();
    _resolveImageSize();
  }

  void _resolveImageSize() {
    final image = FileImage(File(widget.photoPath));
    image.resolve(const ImageConfiguration()).addListener(
      ImageStreamListener((info, _) {
        if (mounted) {
          setState(() {
            _imageSize = Size(
              info.image.width.toDouble(),
              info.image.height.toDouble(),
            );
          });
        }
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        // Compute the actual rendered size after BoxFit.contain
        double renderWidth = constraints.maxWidth;
        double renderHeight = constraints.maxHeight;

        if (_imageSize != null) {
          final imageAspect = _imageSize!.width / _imageSize!.height;
          final viewAspect = constraints.maxWidth / constraints.maxHeight;

          if (imageAspect > viewAspect) {
            // Image is wider — constrained by width
            renderWidth = constraints.maxWidth;
            renderHeight = constraints.maxWidth / imageAspect;
          } else {
            // Image is taller — constrained by height
            renderHeight = constraints.maxHeight;
            renderWidth = constraints.maxHeight * imageAspect;
          }
        }

        return Stack(
          alignment: Alignment.center,
          children: [
            Image.file(
              File(widget.photoPath),
              width: constraints.maxWidth,
              height: constraints.maxHeight,
              fit: BoxFit.contain,
            ),
            BoundingBoxOverlay(
              detections: widget.detections,
              imageWidth: 1,
              imageHeight: 1,
              viewWidth: renderWidth,
              viewHeight: renderHeight,
            ),
          ],
        );
      },
    );
  }
}

class ResultsScreen extends StatelessWidget {
  final String photoPath;
  final DetectionResult result;
  final VoidCallback? onScanAnother;

  const ResultsScreen({
    super.key,
    required this.photoPath,
    required this.result,
    this.onScanAnother,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: _ImageWithOverlay(
                photoPath: photoPath,
                detections: result.detections,
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Text(
                    '${result.count} items detected',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Inference: ${result.inferenceTimeMs.round()}ms',
                    style: const TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF007AFF),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                ),
                onPressed: onScanAnother ?? () => Navigator.pop(context),
                child: const Text(
                  'Scan Another Shelf',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
