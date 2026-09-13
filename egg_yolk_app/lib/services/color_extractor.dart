import 'dart:math' as math;
import 'package:image/image.dart' as img;

/// Service for extracting color features from egg yolk images using
/// Center Circular Masking (42% radius) and converting to CIE D65 CIELAB.
class ColorExtractor {
  /// Reference white point (CIE D65 standard, 2° observer)
  static const double xn = 0.95047;
  static const double yn = 1.00000;
  static const double zn = 1.08883;

  /// Extracts [mean_R, mean_G, mean_B, L*, a*, b*] from [image] using
  /// Center Circular Mask at radius = min(w, h) * 0.42.
  static List<double> extractColorFeatures(img.Image image) {
    final w = image.width;
    final h = image.height;
    final cx = w ~/ 2;
    final cy = h ~/ 2;
    final radius = (math.min(w, h) * 0.42).round();
    final radiusSq = radius * radius;

    double sumR = 0.0;
    double sumG = 0.0;
    double sumB = 0.0;
    int count = 0;

    for (int y = 0; y < h; y++) {
      final dy = y - cy;
      final dySq = dy * dy;
      for (int x = 0; x < w; x++) {
        final dx = x - cx;
        if (dx * dx + dySq <= radiusSq) {
          final pixel = image.getPixel(x, y);
          sumR += pixel.r.toDouble();
          sumG += pixel.g.toDouble();
          sumB += pixel.b.toDouble();
          count++;
        }
      }
    }

    if (count == 0) {
      // Fallback: average across all pixels
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final pixel = image.getPixel(x, y);
          sumR += pixel.r.toDouble();
          sumG += pixel.g.toDouble();
          sumB += pixel.b.toDouble();
          count++;
        }
      }
    }

    final meanR = sumR / count;
    final meanG = sumG / count;
    final meanB = sumB / count;

    final lab = rgbToCielab(meanR, meanG, meanB);
    return [meanR, meanG, meanB, lab[0], lab[1], lab[2]];
  }

  /// Converts RGB (0..255) to CIELAB (L*, a*, b*) using CIE D65 standard.
  static List<double> rgbToCielab(double r, double g, double b) {
    // 1. sRGB gamma correction to linear RGB
    double rLin = r / 255.0;
    double gLin = g / 255.0;
    double bLin = b / 255.0;

    rLin = (rLin > 0.04045)
        ? math.pow((rLin + 0.055) / 1.055, 2.4).toDouble()
        : rLin / 12.92;
    gLin = (gLin > 0.04045)
        ? math.pow((gLin + 0.055) / 1.055, 2.4).toDouble()
        : gLin / 12.92;
    bLin = (bLin > 0.04045)
        ? math.pow((bLin + 0.055) / 1.055, 2.4).toDouble()
        : bLin / 12.92;

    // 2. Linear RGB to XYZ (IEC 61966-2-1 precise matrix — synced with local_predict_service.dart)
    final x = 0.4124564 * rLin + 0.3575761 * gLin + 0.1804375 * bLin;
    final y = 0.2126729 * rLin + 0.7151522 * gLin + 0.0721750 * bLin;
    final z = 0.0193339 * rLin + 0.1191920 * gLin + 0.9503041 * bLin;

    // 3. XYZ to CIELAB
    final xr = x / xn;
    final yr = y / yn;
    final zr = z / zn;

    final fx = (xr > 0.008856)
        ? math.pow(xr, 1.0 / 3.0).toDouble()
        : (7.787 * xr + 16.0 / 116.0);
    final fy = (yr > 0.008856)
        ? math.pow(yr, 1.0 / 3.0).toDouble()
        : (7.787 * yr + 16.0 / 116.0);
    final fz = (zr > 0.008856)
        ? math.pow(zr, 1.0 / 3.0).toDouble()
        : (7.787 * zr + 16.0 / 116.0);

    final l = 116.0 * fy - 16.0;
    final a = 500.0 * (fx - fy);
    final labB = 200.0 * (fy - fz);

    return [l, a, labB];
  }
}
