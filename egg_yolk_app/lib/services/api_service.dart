import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import '../models/prediction_result.dart';

class ApiService {
  // Default URL: http://10.0.2.2:8000 for Android Emulator alias to localhost
  static String baseUrl = 'http://10.0.2.2:8000';

  static Future<PredictionResult> predictImage(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict-image');
    
    final request = http.MultipartRequest('POST', uri);
    
    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        imageFile.path,
      ),
    );

    try {
      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 15),
      );

      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        return PredictionResult.fromJson(data);
      } else {
        try {
          final errorData = json.decode(response.body);
          throw Exception(errorData['detail'] ?? 'Server error: ${response.statusCode}');
        } catch (_) {
          throw Exception('Server error: ${response.statusCode}');
        }
      }
    } catch (e) {
      throw Exception('Connection error: $e');
    }
  }
}
