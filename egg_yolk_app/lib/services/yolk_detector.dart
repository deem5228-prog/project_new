import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Yolk detection and cropping service using pure Dart math matching OpenCV HSV segmentation.
class YolkDetector {
  // HSV thresholds — converted from OpenCV scale
  // OpenCV: H in [0, 180], S in [0, 255], V in [0, 255]
  // Standard Dart: H in [0°, 360°], S in [0.0, 1.0], V in [0.0, 1.0]
  static const double hMin = 20.0;  // was OpenCV H=10  (×2)
  static const double hMax = 56.0;  // was OpenCV H=28  (×2)
  static const double sMin = 0.314; // was OpenCV S=80  (÷255 ≈ 0.3137)
  static const double vMin = 0.314; // was OpenCV V=80  (÷255 ≈ 0.3137)
  static const double vMax = 0.980; // was OpenCV V=250 (÷255 ≈ 0.9804)

  // OpenCV equivalent integer thresholds
  static const int cvHMin = 10;
  static const int cvHMax = 28;
  static const int cvSMin = 80;
  static const int cvSMax = 255;
  static const int cvVMin = 80;
  static const int cvVMax = 250;

  /// Public test helper for yolk color qualification.
  static bool isYolkColor(double r, double g, double b) => _isYolkColor(r, g, b);

  /// Tests whether an RGB pixel (0-255) falls within the yolk HSV range.
  /// Converts RGB to standard HSV (0-360°, 0-1, 0-1) and verifies equivalence
  /// with OpenCV 8-bit scale thresholds.
  static bool _isYolkColor(double r, double g, double b) {
    final maxVal = math.max(r, math.max(g, b));
    final minVal = math.min(r, math.min(g, b));
    final delta = maxVal - minVal;

    // V in [0.0, 1.0]
    final v = maxVal / 255.0;
    // S in [0.0, 1.0]
    final s = maxVal == 0 ? 0.0 : delta / maxVal;

    // H in [0°, 360°]
    double h = 0.0;
    if (delta != 0) {
      if (maxVal == r) {
        h = 60.0 * ((g - b) / delta);
      } else if (maxVal == g) {
        h = 120.0 + 60.0 * ((b - r) / delta);
      } else {
        h = 240.0 + 60.0 * ((r - g) / delta);
      }
      if (h < 0) h += 360.0;
    }

    // OpenCV 8-bit scale conversion:
    // dart_H = opencv_H * 2  =>  opencv_H = round(H / 2)
    // dart_S = opencv_S / 255 =>  opencv_S = round(S * 255)
    // dart_V = opencv_V / 255 =>  opencv_V = round(V * 255)
    final cvH = (h / 2.0).round();
    final cvS = (s * 255.0).round();
    final cvV = (v * 255.0).round();

    return cvH >= cvHMin && cvH <= cvHMax &&
           cvS >= cvSMin && cvS <= cvSMax &&
           cvV >= cvVMin && cvV <= cvVMax;
  }

  /// Detects the egg yolk in [image] using HSV thresholding and centroid/radius estimation.
  /// Returns a Map with coordinates: {'x', 'y', 'width', 'height', 'cx', 'cy', 'radius'}.
  Map<String, int>? detect(img.Image image) {
    final w = image.width;
    final h = image.height;

    final xCoords = <int>[];
    final yCoords = <int>[];

    // Scan pixels (using step=2 for large images to maximize performance)
    final step = (w > 800 || h > 800) ? 2 : 1;

    for (int y = 0; y < h; y += step) {
      for (int x = 0; x < w; x += step) {
        final pixel = image.getPixel(x, y);
        if (_isYolkColor(pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble())) {
          xCoords.add(x);
          yCoords.add(y);
        }
      }
    }

    if (xCoords.isEmpty) {
      // Fallback: central region
      final r = (math.min(w, h) * 0.40).round();
      final cx = w ~/ 2;
      final cy = h ~/ 2;
      final sz = (r * 1.08).round();
      final x1 = math.max(0, cx - sz);
      final y1 = math.max(0, cy - sz);
      final x2 = math.min(w, cx + sz);
      final y2 = math.min(h, cy + sz);
      return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1,
        'cx': cx,
        'cy': cy,
        'radius': r,
      };
    }

    // Median centroid for robust center localization (sort copies to preserve (x,y) pairing)
    final sortedX = List<int>.from(xCoords)..sort();
    final sortedY = List<int>.from(yCoords)..sort();
    final cx = sortedX[sortedX.length ~/ 2];
    final cy = sortedY[sortedY.length ~/ 2];

    // Compute enclosing radius from 99.5th percentile distance
    final dists = Float64List(xCoords.length);
    for (int i = 0; i < xCoords.length; i++) {
      final dx = (xCoords[i] - cx).toDouble();
      final dy = (yCoords[i] - cy).toDouble();
      dists[i] = math.sqrt(dx * dx + dy * dy);
    }
    dists.sort();
    final pIdx = (dists.length * 0.995).floor().clamp(0, dists.length - 1);
    final radius = dists[pIdx].round();

    // 1.08x scale matching Python auto_crop_yolk formula
    final sz = (radius * 1.08).round();
    final x1 = math.max(0, cx - sz);
    final y1 = math.max(0, cy - sz);
    final x2 = math.min(w, cx + sz);
    final y2 = math.min(h, cy + sz);

    return {
      'x': x1,
      'y': y1,
      'width': x2 - x1,
      'height': y2 - y1,
      'cx': cx,
      'cy': cy,
      'radius': radius,
    };
  }

  /// Crops a square egg yolk region from [image] given the [detection] coordinates.
  img.Image? crop(img.Image image, Map<String, int> detection) {
    final x = detection['x'] ?? 0;
    final y = detection['y'] ?? 0;
    final w = detection['width'] ?? (detection['size'] ?? image.width);
    final h = detection['height'] ?? (detection['size'] ?? image.height);

    final clX = x.clamp(0, image.width - 1);
    final clY = y.clamp(0, image.height - 1);
    final clW = w.clamp(1, image.width - clX);
    final clH = h.clamp(1, image.height - clY);

    return img.copyCrop(image, x: clX, y: clY, width: clW, height: clH);
  }
}
