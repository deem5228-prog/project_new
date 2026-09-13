import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:path_provider/path_provider.dart';
import '../services/auto_crop_service.dart';

class CropScreen extends StatefulWidget {
  final File imageFile;

  const CropScreen({super.key, required this.imageFile});

  @override
  State<CropScreen> createState() => _CropScreenState();
}

class _CropScreenState extends State<CropScreen> {
  ui.Image? _decodedImage;
  bool _isProcessing = false;
  bool _isAutoDetecting = false;
  DetectedYolkBox? _detectedBox;

  // Circular Cropper state: Center (cx, cy) and radius (r)
  double _centerX = 150.0;
  double _centerY = 150.0;
  double _radius = 80.0;
  bool _isInitialized = false;

  // Display image geometry within LayoutBuilder
  double _dispL = 0.0;
  double _dispT = 0.0;
  double _dispW = 0.0;
  double _dispH = 0.0;

  // Interaction dragging mode:
  // 'none' | 'center' | 'top' | 'bottom' | 'left' | 'right'
  String _dragMode = 'none';
  Offset _startTouch = Offset.zero;
  double _startCx = 0.0;
  double _startCy = 0.0;
  double _startRadius = 0.0;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  @override
  void dispose() {
    _decodedImage?.dispose();
    super.dispose();
  }

  Future<void> _loadImage() async {
    try {
      final bytes = await widget.imageFile.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      // Run on-device yolk detection in parallel
      final detected = await AutoCropService.detectYolk(widget.imageFile);

      if (mounted) {
        setState(() {
          _decodedImage = frame.image;
          _detectedBox = detected;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('ไม่สามารถโหลดรูปภาพได้: $e', style: GoogleFonts.kanit()),
            backgroundColor: Colors.redAccent,
          ),
        );
        Navigator.pop(context);
      }
    }
  }

  Future<void> _runAutoDetect() async {
    if (_isAutoDetecting || _dispW <= 0 || _dispH <= 0) return;
    setState(() {
      _isAutoDetecting = true;
    });

    try {
      final detected = await AutoCropService.detectYolk(widget.imageFile);
      if (mounted) {
        setState(() {
          _isAutoDetecting = false;
          _detectedBox = detected;
          if (detected.isDetected) {
            final boxW = detected.normWidth * _dispW;
            final boxH = detected.normHeight * _dispH;
            _centerX = _dispL + (detected.normX * _dispW) + (boxW / 2.0);
            _centerY = _dispT + (detected.normY * _dispH) + (boxH / 2.0);
            _radius = (math.min(boxW, boxH) / 2.0) * 0.95;
            _radius = _radius.clamp(30.0, math.min(_dispW, _dispH) / 2.0);

            final minX = _dispL + _radius;
            final maxX = _dispL + _dispW - _radius;
            final minY = _dispT + _radius;
            final maxY = _dispT + _dispH - _radius;

            _centerX = _centerX.clamp(
              minX < maxX ? minX : _dispL + _dispW / 2.0,
              maxX > minX ? maxX : _dispL + _dispW / 2.0,
            );
            _centerY = _centerY.clamp(
              minY < maxY ? minY : _dispT + _dispH / 2.0,
              maxY > minY ? maxY : _dispT + _dispH / 2.0,
            );
          }
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              detected.isDetected
                  ? '🎯 ตรวจจับและล็อกเป้าไข่แดงให้อัตโนมัติแล้ว'
                  : '⚠️ ตรวจจับไม่ชัดเจน ใช้ตำแหน่งเดิม',
              style: GoogleFonts.kanit(),
            ),
            duration: const Duration(seconds: 1),
            backgroundColor: detected.isDetected ? const Color(0xFF10B981) : Colors.orange,
          ),
        );
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _isAutoDetecting = false;
        });
      }
    }
  }

  Future<void> _confirmCrop() async {
    if (_decodedImage == null || _isProcessing || _dispW <= 0 || _dispH <= 0) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      // Scale factors from display coordinates to original natural image resolution.
      // Use average of scaleX and scaleY for radius mapping so the circular crop is
      // correct even when the image is non-square (scaleX ≠ scaleY under BoxFit.contain).
      final double scaleX = _decodedImage!.width / _dispW;
      final double scaleY = _decodedImage!.height / _dispH;
      final double scaleAvg = (scaleX + scaleY) / 2.0;

      final double natCx = (_centerX - _dispL) * scaleX;
      final double natCy = (_centerY - _dispT) * scaleY;
      final double natRadius = _radius * scaleAvg; // use average scale for radius

      // 1.08x bounding box matching OpenCV model training pipeline
      final double cropHalfSize = natRadius * 1.08;
      final double cropSize = cropHalfSize * 2.0;

      double cropX = natCx - cropHalfSize;
      double cropY = natCy - cropHalfSize;

      // Clamp to image bounds
      final double imgW = _decodedImage!.width.toDouble();
      final double imgH = _decodedImage!.height.toDouble();

      cropX = cropX.clamp(0.0, math.max(0.0, imgW - cropSize));
      cropY = cropY.clamp(0.0, math.max(0.0, imgH - cropSize));
      final double finalSize = math.min(cropSize, math.min(imgW - cropX, imgH - cropY));

      // Guard: finalSize must be at least 1 pixel — prevents toImage(0,0) crash
      if (finalSize < 1.0) {
        throw Exception('พื้นที่ crop เล็กเกินไป กรุณาปรับวงกลมให้ใหญ่ขึ้น');
      }

      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, finalSize, finalSize));

      final srcRect = Rect.fromLTWH(cropX, cropY, finalSize, finalSize);
      final dstRect = Rect.fromLTWH(0, 0, finalSize, finalSize);

      canvas.drawImageRect(
        _decodedImage!,
        srcRect,
        dstRect,
        Paint()..filterQuality = FilterQuality.high,
      );

      final picture = recorder.endRecording();
      final croppedImg = await picture.toImage(finalSize.toInt(), finalSize.toInt());

      // Encode before disposing
      final byteData = await croppedImg.toByteData(format: ui.ImageByteFormat.png);

      // Dispose native resources immediately after encoding to prevent memory leaks
      croppedImg.dispose();
      picture.dispose();

      if (byteData == null) throw Exception('Failed to encode cropped image');

      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/cropped_yolk_${DateTime.now().millisecondsSinceEpoch}.png');
      await file.writeAsBytes(byteData.buffer.asUint8List());

      if (mounted) {
        Navigator.pop(context, file);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('เกิดข้อผิดพลาดในการครอบตัด: $e', style: GoogleFonts.kanit()),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF141414),
      body: SafeArea(
        child: Column(
          children: [
            // 1. Top Bar matching Mockup (Back button + "ตรวจสอบตำแหน่ง" + "4 จุดรอบวงกลม")
            _buildTopBar(),

            // 2. Central Interactive Canvas Area
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final canvasW = constraints.maxWidth;
                  final canvasH = constraints.maxHeight;

                  if (!_isInitialized && _decodedImage != null) {
                    final imgAspect = _decodedImage!.width / _decodedImage!.height;
                    final canvasAspect = canvasW / canvasH;

                    double dispW, dispH, dispL, dispT;
                    if (imgAspect > canvasAspect) {
                      dispW = canvasW;
                      dispH = canvasW / imgAspect;
                      dispL = 0.0;
                      dispT = (canvasH - dispH) / 2.0;
                    } else {
                      dispH = canvasH;
                      dispW = canvasH * imgAspect;
                      dispL = (canvasW - dispW) / 2.0;
                      dispT = 0.0;
                    }

                    _dispL = dispL;
                    _dispT = dispT;
                    _dispW = dispW;
                    _dispH = dispH;

                    // Automatically center on detected yolk or use smart center default
                    if (_detectedBox != null && _detectedBox!.isDetected) {
                      final boxW = _detectedBox!.normWidth * dispW;
                      final boxH = _detectedBox!.normHeight * dispH;
                      _centerX = dispL + (_detectedBox!.normX * dispW) + (boxW / 2.0);
                      _centerY = dispT + (_detectedBox!.normY * dispH) + (boxH / 2.0);
                      _radius = (math.min(boxW, boxH) / 2.0) * 0.95;
                    } else {
                      _centerX = dispL + (dispW / 2.0);
                      _centerY = dispT + (dispH / 2.0);
                      _radius = math.min(dispW, dispH) * 0.35;
                    }

                    _radius = _radius.clamp(30.0, math.min(dispW, dispH) / 2.0);

                    // Clamp center so the circle is perfectly within image bounds from the start
                    final minX = dispL + _radius;
                    final maxX = dispL + dispW - _radius;
                    final minY = dispT + _radius;
                    final maxY = dispT + dispH - _radius;

                    _centerX = _centerX.clamp(
                      minX < maxX ? minX : dispL + dispW / 2.0,
                      maxX > minX ? maxX : dispL + dispW / 2.0,
                    );
                    _centerY = _centerY.clamp(
                      minY < maxY ? minY : dispT + dispH / 2.0,
                      maxY > minY ? maxY : dispT + dispH / 2.0,
                    );

                    _isInitialized = true;
                  }

                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      // Base Image
                      if (_decodedImage != null)
                        Center(
                          child: Image.file(
                            widget.imageFile,
                            fit: BoxFit.contain,
                            width: canvasW,
                            height: canvasH,
                          ),
                        )
                      else
                        const Center(
                          child: CircularProgressIndicator(color: Color(0xFFFF9800)),
                        ),

                      // Circle Cutout Mask & Crosshair Painter
                      CustomPaint(
                        painter: _CircleMaskPainter(
                          centerX: _centerX,
                          centerY: _centerY,
                          radius: _radius,
                        ),
                      ),

                      // Left Side Text Guide: "ลากใน = ย้าย"
                      Positioned(
                        left: math.max(12.0, _centerX - _radius - 90.0),
                        top: _centerY - 18.0,
                        child: IgnorePointer(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                            decoration: BoxDecoration(
                              color: const Color(0x77000000),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              'ลากใน\n= ย้าย',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.kanit(
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                color: Colors.white,
                                height: 1.2,
                              ),
                            ),
                          ),
                        ),
                      ),

                      // Right Side Text Guide: "ลากจุดขาว = ปรับขนาด"
                      Positioned(
                        left: math.min(canvasW - 95.0, _centerX + _radius + 12.0),
                        top: _centerY - 18.0,
                        child: IgnorePointer(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                            decoration: BoxDecoration(
                              color: const Color(0x77000000),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              'ลากจุดขาว\n= ปรับขนาด',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.kanit(
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                color: Colors.white,
                                height: 1.2,
                              ),
                            ),
                          ),
                        ),
                      ),

                      // Circle Center Drag Gesture Area
                      Positioned(
                        left: _centerX - _radius,
                        top: _centerY - _radius,
                        width: _radius * 2,
                        height: _radius * 2,
                        child: GestureDetector(
                          behavior: HitTestBehavior.translucent,
                          onPanStart: (details) {
                            setState(() {
                              _dragMode = 'center';
                              _startTouch = details.globalPosition;
                              _startCx = _centerX;
                              _startCy = _centerY;
                            });
                          },
                          onPanUpdate: (details) {
                            if (_dragMode == 'center') {
                              final dx = details.globalPosition.dx - _startTouch.dx;
                              final dy = details.globalPosition.dy - _startTouch.dy;

                              final minX = _dispL + _radius;
                              final maxX = _dispL + _dispW - _radius;
                              final minY = _dispT + _radius;
                              final maxY = _dispT + _dispH - _radius;

                              setState(() {
                                _centerX = (_startCx + dx).clamp(
                                  minX < maxX ? minX : _dispL,
                                  maxX > minX ? maxX : _dispL + _dispW,
                                );
                                _centerY = (_startCy + dy).clamp(
                                  minY < maxY ? minY : _dispT,
                                  maxY > minY ? maxY : _dispT + _dispH,
                                );
                              });
                            }
                          },
                          onPanEnd: (_) => setState(() => _dragMode = 'none'),
                          child: Container(color: Colors.transparent),
                        ),
                      ),

                      // 4 White Handle Dots (Top, Bottom, Left, Right)
                      _buildHandle('top', _centerX, _centerY - _radius),
                      _buildHandle('bottom', _centerX, _centerY + _radius),
                      _buildHandle('left', _centerX - _radius, _centerY),
                      _buildHandle('right', _centerX + _radius, _centerY),
                    ],
                  );
                },
              ),
            ),

            // 3. Bottom Card matching Mockup (White container with hints & 2 action buttons)
            _buildBottomPanel(),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
      color: const Color(0xFF141414),
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Back button
          Align(
            alignment: Alignment.centerLeft,
            child: IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.white, size: 24),
              onPressed: () => Navigator.pop(context),
            ),
          ),
          // Center title & subtitle
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'ตรวจสอบตำแหน่ง',
                style: GoogleFonts.kanit(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                '4 จุดรอบวงกลม',
                style: GoogleFonts.kanit(
                  fontSize: 13,
                  fontWeight: FontWeight.w400,
                  color: const Color(0xFFB0B0B0),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHandle(String type, double x, double y) {
    const double touchHitSize = 48.0;

    return Positioned(
      left: x - (touchHitSize / 2),
      top: y - (touchHitSize / 2),
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onPanStart: (details) {
          setState(() {
            _dragMode = type;
            _startTouch = details.globalPosition;
            _startRadius = _radius;
          });
        },
        onPanUpdate: (details) {
          if (_dragMode == type) {
            final dx = details.globalPosition.dx - _startTouch.dx;
            final dy = details.globalPosition.dy - _startTouch.dy;

            double newR = _startRadius;
            if (type == 'top') {
              newR = _startRadius - dy;
            } else if (type == 'bottom') {
              newR = _startRadius + dy;
            } else if (type == 'left') {
              newR = _startRadius - dx;
            } else if (type == 'right') {
              newR = _startRadius + dx;
            }

            final maxR = math.min(_dispW, _dispH) / 2.0;
            final clampedR = newR.clamp(30.0, maxR);
            final minX = _dispL + clampedR;
            final maxX = _dispL + _dispW - clampedR;
            final minY = _dispT + clampedR;
            final maxY = _dispT + _dispH - clampedR;

            setState(() {
              _radius = clampedR;
              _centerX = _centerX.clamp(
                minX < maxX ? minX : _dispL + _dispW / 2.0,
                maxX > minX ? maxX : _dispL + _dispW / 2.0,
              );
              _centerY = _centerY.clamp(
                minY < maxY ? minY : _dispT + _dispH / 2.0,
                maxY > minY ? maxY : _dispT + _dispH / 2.0,
              );
            });
          }
        },
        onPanEnd: (_) => setState(() => _dragMode = 'none'),
        child: Container(
          width: touchHitSize,
          height: touchHitSize,
          alignment: Alignment.center,
          color: Colors.transparent,
          child: Container(
            width: 20,
            height: 20,
            decoration: BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
              border: Border.all(color: const Color(0xFFFF9800), width: 2.5),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x66000000),
                  blurRadius: 4,
                  offset: Offset(0, 2),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        boxShadow: [
          BoxShadow(
            color: Color(0x22000000),
            blurRadius: 10,
            offset: Offset(0, -2),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Instruction texts matching Mockup
          Text(
            'ลากในวงกลม → ย้ายตำแหน่ง',
            style: GoogleFonts.kanit(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: const Color(0xFF4B5563),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            'ลากจุดขาว → ปรับขนาด',
            style: GoogleFonts.kanit(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: const Color(0xFF4B5563),
            ),
          ),
          const SizedBox(height: 16),

          // Two Action Buttons: "ตรวจจับใหม่" & "ยืนยันตำแหน่ง"
          Row(
            children: [
              // Button 1: "ตรวจจับใหม่" (Outlined)
              Expanded(
                child: SizedBox(
                  height: 48,
                  child: OutlinedButton(
                    onPressed: _isAutoDetecting ? null : _runAutoDetect,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: const Color(0xFF1F2937),
                      backgroundColor: Colors.white,
                      side: const BorderSide(color: Color(0xFFD1D5DB), width: 1.5),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: _isAutoDetecting
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              color: Color(0xFFFF9800),
                              strokeWidth: 2,
                            ),
                          )
                        : Text(
                            'ตรวจจับใหม่',
                            style: GoogleFonts.kanit(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                  ),
                ),
              ),
              const SizedBox(width: 14),

              // Button 2: "ยืนยันตำแหน่ง" (Filled Vibrant Orange)
              Expanded(
                child: SizedBox(
                  height: 48,
                  child: ElevatedButton(
                    onPressed: _isProcessing ? null : _confirmCrop,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFFF9800),
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: _isProcessing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : Text(
                            'ยืนยันตำแหน่ง',
                            style: GoogleFonts.kanit(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Custom Painter for Circular Mask Overlay with center crosshair & axis dashed lines
class _CircleMaskPainter extends CustomPainter {
  final double centerX;
  final double centerY;
  final double radius;

  _CircleMaskPainter({
    required this.centerX,
    required this.centerY,
    required this.radius,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Dark overlay outside the circle
    final backgroundPath = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final circlePath = Path()
      ..addOval(Rect.fromCircle(center: Offset(centerX, centerY), radius: radius));
    final maskPath = Path.combine(PathOperation.difference, backgroundPath, circlePath);

    final darkPaint = Paint()
      ..color = const Color(0xA6000000) // ~65% opacity black
      ..style = PaintingStyle.fill;
    canvas.drawPath(maskPath, darkPaint);

    // 2. Circle Border (Warm Orange)
    final borderPaint = Paint()
      ..color = const Color(0xFFFF9800)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawCircle(Offset(centerX, centerY), radius, borderPaint);

    // 3. Dashed Axis Guide Lines
    final dashedPaint = Paint()
      ..color = const Color(0x66FFFFFF) // ~40% white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    _drawDashedLine(
      canvas,
      Offset(centerX - radius, centerY),
      Offset(centerX + radius, centerY),
      dashedPaint,
    );
    _drawDashedLine(
      canvas,
      Offset(centerX, centerY - radius),
      Offset(centerX, centerY + radius),
      dashedPaint,
    );

    // 4. Center Crosshair '+' (Orange)
    final crossPaint = Paint()
      ..color = const Color(0xFFFF9800)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    const double crossSize = 10.0;
    canvas.drawLine(
      Offset(centerX - crossSize, centerY),
      Offset(centerX + crossSize, centerY),
      crossPaint,
    );
    canvas.drawLine(
      Offset(centerX, centerY - crossSize),
      Offset(centerX, centerY + crossSize),
      crossPaint,
    );
  }

  void _drawDashedLine(Canvas canvas, Offset p1, Offset p2, Paint paint) {
    const double dashWidth = 5.0;
    const double dashSpace = 4.0;

    final double dx = p2.dx - p1.dx;
    final double dy = p2.dy - p1.dy;
    final double distance = math.sqrt(dx * dx + dy * dy);
    if (distance <= 0) return;

    final double unitX = dx / distance;
    final double unitY = dy / distance;

    double currentDist = 0.0;
    while (currentDist < distance) {
      final double nextDist = math.min(currentDist + dashWidth, distance);
      canvas.drawLine(
        Offset(p1.dx + unitX * currentDist, p1.dy + unitY * currentDist),
        Offset(p1.dx + unitX * nextDist, p1.dy + unitY * nextDist),
        paint,
      );
      currentDist += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant _CircleMaskPainter oldDelegate) {
    return oldDelegate.centerX != centerX ||
        oldDelegate.centerY != centerY ||
        oldDelegate.radius != radius;
  }
}
