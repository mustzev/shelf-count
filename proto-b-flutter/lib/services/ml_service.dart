import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';
import 'postprocess.dart';

const _modelAsset = 'assets/models/yolov8m-sku110k.tflite';
const _inputSize = 640;
const _confidenceThreshold = 0.25;
const _iouThreshold = 0.5;

// YOLOv8 output dimensions
const _numBoxes = 8400;
const _numValues = 5; // [cx, cy, w, h, confidence]

class MlService {
  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(_modelAsset);
  }

  /// Run inference on a JPEG/PNG image.
  ///
  /// [imageBytes] — raw file bytes (not decoded pixels).
  /// Returns null if the model is not loaded or image decode fails.
  DetectionResult? runInference(Uint8List imageBytes) {
    if (_interpreter == null) return null;

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return null;

    // Resize to 640x640
    final resized = img.copyResize(decoded, width: _inputSize, height: _inputSize);

    // Convert to [1, 640, 640, 3] float32 input tensor with pixels normalized to 0–1
    final input = _imageToInputTensor(resized);

    // Prepare output buffer — shape must match [1, 5, 8400] as nested lists.
    // tflite_flutter may replace inner list references on copyTo, so we
    // read results from outputs[0] after inference, not from the local variable.
    final outputs = <int, Object>{
      0: List.generate(1, (_) =>
          List.generate(_numValues, (_) => List<double>.filled(_numBoxes, 0.0))),
    };

    final stopwatch = Stopwatch()..start();
    _interpreter!.runForMultipleInputs([input], outputs);
    stopwatch.stop();

    // Read from outputs map — tflite_flutter may have replaced references.
    // Flatten [1, 5, 8400] to row-major [5 * 8400] for processYoloOutputs.
    final nested = (outputs[0] as List)[0] as List; // [5, 8400]
    final rawOutput = <double>[];
    for (final row in nested) {
      for (final v in row as List) {
        rawOutput.add((v as num).toDouble());
      }
    }

    final detections = processYoloOutputs(
      rawOutput: rawOutput,
      numBoxes: _numBoxes,
      numValues: _numValues,
      label: 'object',
      scoreThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
    );

    return DetectionResult(
      detections: detections,
      count: detections.length,
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000.0,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Convert an image to [1, 640, 640, 3] float32 tensor with pixels
  /// normalized to 0–1 by dividing by 255.0.
  List<List<List<List<double>>>> _imageToInputTensor(img.Image image) {
    return [
      List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              pixel.r / 255.0,
              pixel.g / 255.0,
              pixel.b / 255.0,
            ];
          },
        ),
      ),
    ];
  }
}
