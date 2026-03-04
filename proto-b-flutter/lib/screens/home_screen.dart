import 'package:flutter/material.dart';

import '../models/detection.dart';
import 'camera_screen.dart';
import 'results_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  String? _lastPhotoPath;
  DetectionResult? _lastResult;

  void _onCaptureComplete(String photoPath, DetectionResult result) {
    setState(() {
      _lastPhotoPath = photoPath;
      _lastResult = result;
      _currentIndex = 1;
    });
  }

  void _onScanAnother() {
    setState(() {
      _currentIndex = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          CameraScreen(onCaptureComplete: _onCaptureComplete),
          _lastPhotoPath != null && _lastResult != null
              ? ResultsScreen(
                  photoPath: _lastPhotoPath!,
                  result: _lastResult!,
                  onScanAnother: _onScanAnother,
                )
              : const Center(
                  child: Text(
                    'No results yet.\nCapture a shelf first.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey, fontSize: 16),
                  ),
                ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        backgroundColor: Colors.black,
        selectedItemColor: Colors.white,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: 'Camera',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.list),
            label: 'Results',
          ),
        ],
      ),
    );
  }
}
