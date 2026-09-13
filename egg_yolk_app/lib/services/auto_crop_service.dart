import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

class DetectedYolkBox {
  final double normX; // 0.0 to 1.0 (Top-Left X relative to image width)
  final double normY; // 0.0 to 1.0 (Top-Left Y relative to image height)
  final double normWidth; // 0.0 to 1.0 (Box width relative to image width)
  final double normHeight; // 0.0 to 1.0 (Box height relative to image height)
  final bool isDetected;

  DetectedYolkBox({
    required this.normX,
    required this.normY,
    required this.normWidth,
    required this.normHeight,
    required this.isDetected,
  });
}

class AutoCropService {
  /// Detects the egg yolk in an image using downsampled HSV masking & Largest Blob CCA.
  /// Returns normalized coordinates [0.0 - 1.0] ready for canvas display.
  static Future<DetectedYolkBox> detectYolk(File imageFile) async {
    try {
      final bytes = await imageFile.readAsBytes();

      // 1. Fast decode with downsampling to ~240px for instant analysis (< 20ms)
      const targetDim = 240;
      final codec = await ui.instantiateImageCodec(
        bytes,
        targetWidth: targetDim,
      );
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final w = image.width;
      final h = image.height;

      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      image.dispose();

      if (byteData == null) {
        return _fallbackBox();
      }

      final buffer = byteData.buffer.asUint8List();
      final mask = Uint8List(w * h);

      // 2. HSV Color Segmentation
      for (int i = 0; i < w * h; i++) {
        final offset = i * 4;
        final r = buffer[offset] / 255.0;
        final g = buffer[offset + 1] / 255.0;
        final b = buffer[offset + 2] / 255.0;

        final maxVal = math.max(r, math.max(g, b));
        final minVal = math.min(r, math.min(g, b));
        final delta = maxVal - minVal;

        double hue = 0.0;
        if (delta > 0.0001) {
          if (maxVal == r) {
            hue = 60.0 * (((g - b) / delta) % 6.0);
          } else if (maxVal == g) {
            hue = 60.0 * (((b - r) / delta) + 2.0);
          } else {
            hue = 60.0 * (((r - g) / delta) + 4.0);
          }
          if (hue < 0.0) hue += 360.0;
        }

        final sat = (maxVal > 0.0) ? (delta / maxVal) : 0.0;
        final val = maxVal;

        // Scientific thresholds from 647 dataset EDA:
        // Hue: 5.0 - 55.0 deg, Saturation >= 0.25, Value >= 0.20
        if (hue >= 5.0 && hue <= 55.0 && sat >= 0.25 && val >= 0.20) {
          mask[i] = 1;
        }
      }

      // 3. Connected Component Analysis (BFS Flood-Fill) to find Largest Blob
      final visited = Uint8List(w * h);
      int largestArea = 0;
      int bestMinX = 0, bestMaxX = 0, bestMinY = 0, bestMaxY = 0;
      double bestCx = w / 2.0, bestCy = h / 2.0;

      final queue = <int>[];

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final idx = y * w + x;
          if (mask[idx] == 1 && visited[idx] == 0) {
            // New component
            int area = 0;
            int sumX = 0, sumY = 0;
            int minX = x, maxX = x, minY = y, maxY = y;

            queue.add(idx);
            visited[idx] = 1;

            while (queue.isNotEmpty) {
              final curr = queue.removeLast();
              final cy = curr ~/ w;
              final cx = curr % w;

              area++;
              sumX += cx;
              sumY += cy;
              if (cx < minX) minX = cx;
              if (cx > maxX) maxX = cx;
              if (cy < minY) minY = cy;
              if (cy > maxY) maxY = cy;

              // 4-directional neighbors
              const dx = [1, -1, 0, 0];
              const dy = [0, 0, 1, -1];
              for (int d = 0; d < 4; d++) {
                final nx = cx + dx[d];
                final ny = cy + dy[d];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                  final nidx = ny * w + nx;
                  if (mask[nidx] == 1 && visited[nidx] == 0) {
                    visited[nidx] = 1;
                    queue.add(nidx);
                  }
                }
              }
            }

            // Compare blob area (Largest Blob Selection)
            if (area > largestArea) {
              largestArea = area;
              bestMinX = minX;
              bestMaxX = maxX;
              bestMinY = minY;
              bestMaxY = maxY;
              bestCx = sumX / area;
              bestCy = sumY / area;
            }
          }
        }
      }

      // 4. Validate if significant yolk area was found (> 50 pixels)
      if (largestArea > 50) {
        final bboxW = (bestMaxX - bestMinX).toDouble();
        final bboxH = (bestMaxY - bestMinY).toDouble();

        // Box size with 5% padding
        final boxSize = math.max(bboxW, bboxH) * 1.05;
        final halfSize = boxSize / 2.0;

        // Bounding box top-left
        double x1 = bestCx - halfSize;
        double y1 = bestCy - halfSize;

        // Clamp to image dimensions
        x1 = x1.clamp(0.0, math.max(0.0, w - boxSize));
        y1 = y1.clamp(0.0, math.max(0.0, h - boxSize));

        return DetectedYolkBox(
          normX: x1 / w,
          normY: y1 / h,
          normWidth: boxSize / w,
          normHeight: boxSize / h,
          isDetected: true,
        );
      }

      return _fallbackBox();
    } catch (_) {
      return _fallbackBox();
    }
  }

  static DetectedYolkBox _fallbackBox() {
    return DetectedYolkBox(
      normX: 0.1,
      normY: 0.1,
      normWidth: 0.8,
      normHeight: 0.8,
      isDetected: false,
    );
  }
}
