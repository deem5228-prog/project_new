import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:egg_yolk_app/services/yolk_detector.dart';
import 'package:egg_yolk_app/services/color_extractor.dart';
import 'package:egg_yolk_app/services/local_predict_service.dart';

void main() {
  group('YolkDetector & ColorExtractor Unit Tests', () {
    test('HSV threshold qualification identifies yolk and rejects background', () {
      // Yolk colors within V <= 250 limit
      expect(YolkDetector.isYolkColor(240, 160, 30), isTrue);
      expect(YolkDetector.isYolkColor(200, 100, 20), isTrue);
      expect(YolkDetector.isYolkColor(235, 145, 25), isTrue);

      // Value > 250 is rejected by OpenCV upper limit [28, 255, 250]
      expect(YolkDetector.isYolkColor(255, 180, 50), isFalse);

      // Background colors (egg white, bowl, dark surface)
      expect(YolkDetector.isYolkColor(245, 245, 245), isFalse); // White plate
      expect(YolkDetector.isYolkColor(20, 20, 20), isFalse);     // Dark shadow
      expect(YolkDetector.isYolkColor(40, 80, 220), isFalse);    // Blue background
      expect(YolkDetector.isYolkColor(50, 200, 50), isFalse);    // Green
    });

    test('ColorExtractor CIELAB matches CIE D65 reference values', () {
      // Reference: Python skimage.color.rgb2lab([[[255,180,50]]]/255.0)
      // Matrix now synced with local_predict_service.dart (IEC 61966-2-1)
      final lab = ColorExtractor.rgbToCielab(255, 180, 50);
      expect(lab[0], closeTo(78.542, 0.01));
      expect(lab[1], closeTo(16.933, 0.01));
      expect(lab[2], closeTo(71.568, 0.01));
    });

    // ─── PRODUCTION PATH TESTS ────────────────────────────────────────────────
    // These tests validate LocalPredictService — the actual class used for
    // on-device prediction. ColorExtractor tests above are kept for regression,
    // but these are the critical tests for production correctness.

    test('LocalPredictService CIELAB matches Python skimage reference values', () {
      // Reference: Python skimage.color.rgb2lab([[[255,180,50]]]/255.0)
      // L*=78.542, a*=16.933, b*=71.568
      final lab = LocalPredictService.rgbToCielab(255.0, 180.0, 50.0);
      expect(lab['l']!, closeTo(78.54, 0.1));
      expect(lab['a']!, closeTo(16.93, 0.1));
      expect(lab['b']!, closeTo(71.57, 0.1));
    });

    test('LocalPredictService CIELAB — dark yolk reference RGB(180, 110, 20)', () {
      // Reference: Python skimage.color.rgb2lab([[[180,110,20]]]/255.0)
      // L*=51.91, a*=17.01, b*=58.22
      final lab = LocalPredictService.rgbToCielab(180.0, 110.0, 20.0);
      expect(lab['l']!, closeTo(51.91, 0.1));
      expect(lab['a']!, closeTo(17.01, 0.1));
      expect(lab['b']!, closeTo(58.22, 0.1));
    });

    test('LocalPredictService and ColorExtractor produce identical CIELAB values', () {
      // Both files now use the same IEC 61966-2-1 matrix — this test enforces that.
      const testR = 220.0;
      const testG = 140.0;
      const testB = 35.0;

      final labLocal = LocalPredictService.rgbToCielab(testR, testG, testB);
      final labExtractor = ColorExtractor.rgbToCielab(testR, testG, testB);

      // Tolerance 0.05 — any divergence here means matrices are out of sync again
      expect(labLocal['l']!, closeTo(labExtractor[0], 0.05));
      expect(labLocal['a']!, closeTo(labExtractor[1], 0.05));
      expect(labLocal['b']!, closeTo(labExtractor[2], 0.05));
    });

    test('YolkDetector detect and crop on synthetic image', () {
      final detector = YolkDetector();
      final image = img.Image(width: 200, height: 200);
      img.fill(image, color: img.ColorRgb8(240, 240, 240)); // White background

      // Draw central yolk circle (radius 50, center 100, 100)
      for (int y = 0; y < 200; y++) {
        for (int x = 0; x < 200; x++) {
          final dx = x - 100;
          final dy = y - 100;
          if (dx * dx + dy * dy <= 50 * 50) {
            image.setPixel(x, y, img.ColorRgb8(220, 130, 20)); // Yolk color
          }
        }
      }

      final detection = detector.detect(image);
      expect(detection, isNotNull);
      expect(detection!['cx'], closeTo(100, 2));
      expect(detection['cy'], closeTo(100, 2));

      final cropped = detector.crop(image, detection);
      expect(cropped, isNotNull);
      expect(cropped!.width, equals(detection['width']));
      expect(cropped.height, equals(detection['height']));

      final features = ColorExtractor.extractColorFeatures(cropped);
      expect(features.length, equals(6));
      expect(features[0], closeTo(220.0, 5.0)); // Mean R
      expect(features[1], closeTo(130.0, 5.0)); // Mean G
      expect(features[2], closeTo(20.0, 5.0));  // Mean B
    });
  });
}
