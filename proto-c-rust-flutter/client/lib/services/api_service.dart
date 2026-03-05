import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../config.dart';
import '../models/detection.dart';

class ApiService {
  Future<DetectionResult> analyze(Uint8List imageBytes) async {
    final uri = Uri.parse('${Config.serverUrl}/analyze');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: 'photo.jpg'));

    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 60),
      onTimeout: () => throw Exception('Server unreachable (timeout)'),
    );

    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      final body = jsonDecode(response.body);
      throw Exception(body['error'] ?? 'Server error ${response.statusCode}');
    }

    return DetectionResult.fromJson(jsonDecode(response.body));
  }

  Future<bool> healthCheck() async {
    try {
      final uri = Uri.parse('${Config.serverUrl}/health');
      final response = await http.get(uri).timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
