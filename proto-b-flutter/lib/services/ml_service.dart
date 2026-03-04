import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';
import 'labels.dart';
import 'postprocess.dart';

const _modelAsset = 'assets/models/efficientdet-lite0.tflite';
const _inputSize = 320;
const _confidenceThreshold = 0.4;
const _maxDetections = 25;

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

    // Resize to 320x320
    final resized = img.copyResize(decoded, width: _inputSize, height: _inputSize);

    // Convert to [1, 320, 320, 3] uint8 input tensor
    final input = _imageToInputTensor(resized);

    // Prepare output buffers — shapes must match model outputs exactly.
    // tflite_flutter replaces inner list references on copyTo, so we
    // read results from the outputs map, not from these variables.
    final outputs = <int, Object>{
      0: List.generate(1, (_) => List.generate(_maxDetections, (_) => List<double>.filled(4, 0))),
      1: List.generate(1, (_) => List<double>.filled(_maxDetections, 0)),
      2: List.generate(1, (_) => List<double>.filled(_maxDetections, 0)),
      3: List<double>.filled(1, 0),
    };

    final stopwatch = Stopwatch()..start();
    _interpreter!.runForMultipleInputs([input], outputs);
    stopwatch.stop();

    // Read results from outputs map (inner references may have been replaced)
    final outBoxesRaw = (outputs[0] as List)[0] as List;
    final outClasses = (outputs[1] as List)[0] as List;
    final outScores = (outputs[2] as List)[0] as List;
    final outCount = (outputs[3] as List);

    final count = (outCount[0] as num).round();

    // Flatten boxes from nested list to flat List<double>
    final flatBoxes = <double>[];
    for (var i = 0; i < count; i++) {
      final box = outBoxesRaw[i] as List;
      for (final v in box) {
        flatBoxes.add((v as num).toDouble());
      }
    }

    // Convert to List<double>
    final scores = outScores.take(count).map((v) => (v as num).toDouble()).toList();
    final classes = outClasses.take(count).map((v) => (v as num).toDouble()).toList();

    final detections = processPostNmsOutputs(
      boxes: flatBoxes,
      classes: classes,
      scores: scores,
      count: count,
      labels: coco90Labels,
      scoreThreshold: _confidenceThreshold,
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

  /// Convert an image to [1, 320, 320, 3] uint8 tensor.
  List<List<List<List<int>>>> _imageToInputTensor(img.Image image) {
    return [
      List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt()];
          },
        ),
      ),
    ];
  }
}
