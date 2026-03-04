import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

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
