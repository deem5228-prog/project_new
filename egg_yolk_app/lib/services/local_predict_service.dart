import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/services.dart' show rootBundle;
import '../models/prediction_result.dart';

class LocalPredictService {
  static Map<String, dynamic>? _weightsData;
  static List<double>? _scalerMean;
  static List<double>? _scalerScale;
  static double? _gamma;
  static double? _intercept;
  static List<double>? _dualCoef;
  static List<List<double>>? _supportVectors;
  static bool _isLoaded = false;

  /// Loads the mathematical model parameters from assets/model_weights.json
  static Future<void> loadModel() async {
    if (_isLoaded) return;

    final jsonString = await rootBundle.loadString('assets/model_weights.json');
    _weightsData = json.decode(jsonString) as Map<String, dynamic>;

    _scalerMean = (_weightsData!['scaler_mean'] as List)
        .map((e) => (e as num).toDouble())
        .toList();
    _scalerScale = (_weightsData!['scaler_scale'] as List)
        .map((e) => (e as num).toDouble())
        .toList();
    _gamma = (_weightsData!['gamma'] as num).toDouble();
    _intercept = (_weightsData!['intercept'] as num).toDouble();
    _dualCoef = (_weightsData!['dual_coef'] as List)
        .map((e) => (e as num).toDouble())
        .toList();

    _supportVectors = (_weightsData!['support_vectors'] as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();

    _isLoaded = true;
  }

  /// Converts an sRGB channel (0-255) to linear light for CIELAB
  static double _srgbToLinear(double c) {
    final v = c / 255.0;
    return (v <= 0.04045) ? (v / 12.92) : math.pow((v + 0.055) / 1.055, 2.4).toDouble();
  }

  /// Converts RGB (0-255) to CIELAB (D65 illuminant, matching skimage rgb2lab)
  static Map<String, double> rgbToCielab(double r, double g, double b) {
    final rLin = _srgbToLinear(r);
    final gLin = _srgbToLinear(g);
    final bLin = _srgbToLinear(b);

    // Linear sRGB to XYZ (D65)
    final x = (0.4124564 * rLin + 0.3575761 * gLin + 0.1804375 * bLin) / 0.95047;
    final y = (0.2126729 * rLin + 0.7151522 * gLin + 0.0721750 * bLin) / 1.00000;
    final z = (0.0193339 * rLin + 0.1191920 * gLin + 0.9503041 * bLin) / 1.08883;

    const delta = 6.0 / 29.0;
    const deltaCubed = delta * delta * delta;

    double f(double t) {
      return (t > deltaCubed) ? math.pow(t, 1.0 / 3.0).toDouble() : (t / (3.0 * delta * delta)) + (4.0 / 29.0);
    }

    final fx = f(x);
    final fy = f(y);
    final fz = f(z);

    final l = (116.0 * fy) - 16.0;
    final a = 500.0 * (fx - fy);
    final bLab = 200.0 * (fy - fz);

    return {
      'l': double.parse(l.toStringAsFixed(2)),
      'a': double.parse(a.toStringAsFixed(2)),
      'b': double.parse(bLab.toStringAsFixed(2)),
    };
  }

  /// Extracts average RGB from a cropped yolk image using Center Circular Mask (R=42%)
  static Future<Map<String, double>> extractMeanRgb(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final w = image.width;
    final h = image.height;

    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();

    if (byteData == null) {
      throw Exception('Failed to decode image pixels for color extraction');
    }

    final buffer = byteData.buffer.asUint8List();
    final cx = w ~/ 2;
    final cy = h ~/ 2;
    final radius = (math.min(w, h) * 0.42).round();
    final radiusSq = radius * radius;

    double sumR = 0;
    double sumG = 0;
    double sumB = 0;
    int count = 0;

    for (int y = 0; y < h; y++) {
      final dy = y - cy;
      final dySq = dy * dy;
      for (int x = 0; x < w; x++) {
        final dx = x - cx;
        if (dx * dx + dySq <= radiusSq) {
          final idx = (y * w + x) * 4;
          sumR += buffer[idx];
          sumG += buffer[idx + 1];
          sumB += buffer[idx + 2];
          count++;
        }
      }
    }

    if (count == 0) {
      // Fallback to full image mean if mask is empty
      for (int i = 0; i < w * h; i++) {
        final idx = i * 4;
        sumR += buffer[idx];
        sumG += buffer[idx + 1];
        sumB += buffer[idx + 2];
      }
      count = w * h;
    }

    return {
      'r': double.parse((sumR / count).toStringAsFixed(2)),
      'g': double.parse((sumG / count).toStringAsFixed(2)),
      'b': double.parse((sumB / count).toStringAsFixed(2)),
    };
  }

  /// Performs full On-Device prediction on a cropped yolk image
  static Future<PredictionResult> predictImage(File croppedImageFile) async {
    // 1. Ensure weights are loaded
    await loadModel();

    // 2. Extract 42% circular mask RGB
    final rgb = await extractMeanRgb(croppedImageFile);
    final r = rgb['r']!;
    final g = rgb['g']!;
    final b = rgb['b']!;

    // 3. Convert to CIELAB
    final lab = rgbToCielab(r, g, b);
    final l = lab['l']!;
    final a = lab['a']!;
    final bLab = lab['b']!;

    // 4. Standardize 6 features [r, g, b, l, a, b_lab] with scaler
    final rawFeatures = [r, g, b, l, a, bLab];
    final normFeatures = List<double>.generate(6, (i) {
      return (rawFeatures[i] - _scalerMean![i]) / _scalerScale![i];
    });

    // 5. Evaluate SVR Dual Formula with RBF Kernel
    double rawScore = _intercept!;
    final numSVs = _supportVectors!.length;
    final gamma = _gamma!;

    for (int i = 0; i < numSVs; i++) {
      final sv = _supportVectors![i];
      double distSq = 0.0;
      for (int j = 0; j < 6; j++) {
        final diff = normFeatures[j] - sv[j];
        distSq += diff * diff;
      }
      final rbf = math.exp(-gamma * distSq);
      rawScore += _dualCoef![i] * rbf;
    }

    // 6. Round and clamp between 1 and 15 (matches Python: max(1.0, min(15.0, raw_score)))
    final predictedScore = rawScore.round().clamp(1, 15);

    // Compute chroma and hue angle
    final chroma = math.sqrt(a * a + bLab * bLab);
    double hueAngle = math.atan2(bLab, a) * 180.0 / math.pi;
    if (hueAngle < 0.0) hueAngle += 360.0;

    return PredictionResult(
      predictedScore: predictedScore,
      rawScore: double.parse(rawScore.toStringAsFixed(2)),
      rgb: RGBColor(r: r, g: g, b: b),
      cielab: CIELABColor(
        l: l,
        a: a,
        b: bLab,
        chroma: double.parse(chroma.toStringAsFixed(2)),
        hueAngle: double.parse(hueAngle.toStringAsFixed(2)),
      ),
    );
  }
}
