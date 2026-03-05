import 'package:flutter/material.dart';

import 'models/detection.dart';
import 'screens/camera_screen.dart';
import 'screens/results_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ShelfCountApp());
}

class ShelfCountApp extends StatelessWidget {
  const ShelfCountApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shelf Count',
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? _photoPath;
  DetectionResult? _result;

  void _onCaptureComplete(String photoPath, DetectionResult result) {
    setState(() {
      _photoPath = photoPath;
      _result = result;
    });
  }

  void _onError(String error) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(error), backgroundColor: Colors.red),
    );
  }

  void _onScanAnother() {
    setState(() {
      _photoPath = null;
      _result = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_photoPath != null && _result != null) {
      return ResultsScreen(
        photoPath: _photoPath!,
        result: _result!,
        onScanAnother: _onScanAnother,
      );
    }

    return CameraScreen(
      onCaptureComplete: _onCaptureComplete,
      onError: _onError,
    );
  }
}
