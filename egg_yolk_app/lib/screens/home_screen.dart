import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

import '../services/api_service.dart';
import '../services/local_predict_service.dart';
import '../models/prediction_result.dart';
import 'crop_screen.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;
  bool _useLocalModel = true; // Default to 100% On-Device AI inference
  final TextEditingController _urlController = TextEditingController(text: ApiService.baseUrl);

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _pickAndProcessImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 90,
      );

      if (pickedFile == null) return;

      if (!mounted) return;

      final File originalFile = File(pickedFile.path);

      // Crop specifically egg yolk area with Custom Circular Cropper
      final File? croppedFile = await Navigator.push<File?>(
        context,
        MaterialPageRoute(
          builder: (_) => CropScreen(imageFile: originalFile),
        ),
      );

      if (croppedFile == null) return;

      if (!mounted) return;
      setState(() {
        _isLoading = true;
      });

      // Predict using On-Device Local AI (or Backend API if chosen in settings)
      final PredictionResult result = _useLocalModel
          ? await LocalPredictService.predictImage(croppedFile)
          : await ApiService.predictImage(croppedFile);

      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });

      // Navigate to ResultScreen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            imageFile: croppedFile,
            result: result,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('เกิดข้อผิดพลาด: ${e.toString().replaceAll('Exception: ', '')}', style: GoogleFonts.kanit()),
          backgroundColor: Colors.redAccent,
        ),
      );
    }
  }

  void _showSettingsDialog() {
    bool tempUseLocal = _useLocalModel;
    showDialog(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              backgroundColor: Colors.white,
              title: Text(
                'ตั้งค่าระบบประมวลผล',
                style: GoogleFonts.kanit(color: const Color(0xFF1A1A1A), fontWeight: FontWeight.bold),
              ),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Mode Switch Card
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: tempUseLocal ? const Color(0xFFECFDF5) : const Color(0xFFF8FAFC),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: tempUseLocal ? const Color(0xFF10B981) : const Color(0xFFE2E8F0)),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            tempUseLocal ? Icons.bolt : Icons.cloud_outlined,
                            color: tempUseLocal ? const Color(0xFF059669) : Colors.blueGrey,
                            size: 28,
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  tempUseLocal ? 'On-Device AI (ออฟไลน์)' : 'Backend API (คลาวด์)',
                                  style: GoogleFonts.kanit(fontWeight: FontWeight.bold, fontSize: 13),
                                ),
                                Text(
                                  tempUseLocal ? 'คำนวณในมือถือ ไม่ต้องต่อเน็ต' : 'ยิงวิเคราะห์ผ่านเซิร์ฟเวอร์ FastAPI',
                                  style: GoogleFonts.kanit(fontSize: 11, color: Colors.black54),
                                ),
                              ],
                            ),
                          ),
                          Switch(
                            value: tempUseLocal,
                            activeTrackColor: const Color(0xFF10B981),
                            onChanged: (val) {
                              setDialogState(() {
                                tempUseLocal = val;
                              });
                            },
                          ),
                        ],
                      ),
                    ),
                    if (!tempUseLocal) ...[
                      const SizedBox(height: 16),
                      TextField(
                        controller: _urlController,
                        style: GoogleFonts.outfit(color: const Color(0xFF1A1A1A)),
                        decoration: InputDecoration(
                          labelText: 'API Base URL',
                          labelStyle: GoogleFonts.kanit(color: Colors.black54),
                          hintText: 'http://10.0.2.2:8000',
                          enabledBorder: const OutlineInputBorder(
                            borderSide: BorderSide(color: Color(0xFF1A1A1A)),
                          ),
                          focusedBorder: const OutlineInputBorder(
                            borderSide: BorderSide(color: Color(0xFFFFC93C), width: 2),
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text('ยกเลิก', style: GoogleFonts.kanit(color: Colors.black54)),
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF1A1A1A)),
                  onPressed: () {
                    setState(() {
                      _useLocalModel = tempUseLocal;
                      ApiService.baseUrl = _urlController.text.trim();
                    });
                    // Capture messenger BEFORE pop — dialog context becomes invalid after pop
                    final messenger = ScaffoldMessenger.of(context);
                    Navigator.pop(context);
                    messenger.showSnackBar(
                      SnackBar(
                        content: Text(
                          _useLocalModel
                              ? 'สลับเป็นโหมด On-Device AI (ออฟไลน์ 100%) เรียบร้อย'
                              : 'บันทึก URL ใหม่เรียบร้อยแล้ว: ${ApiService.baseUrl}',
                          style: GoogleFonts.kanit(),
                        ),
                        backgroundColor: Colors.green,
                      ),
                    );
                  },
                  child: Text('บันทึก', style: GoogleFonts.kanit(color: Colors.white)),
                ),
              ],
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text(
          'วิธีใช้',
          style: GoogleFonts.kanit(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: const Color(0xFF1A1A1A),
          ),
        ),
        actions: [
          Container(
            margin: const EdgeInsets.symmetric(vertical: 13),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: _useLocalModel ? const Color(0xFFDCFCE7) : const Color(0xFFEFF6FF),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _useLocalModel ? const Color(0xFF10B981) : const Color(0xFF3B82F6)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _useLocalModel ? Icons.bolt : Icons.cloud,
                  size: 13,
                  color: _useLocalModel ? const Color(0xFF15803D) : const Color(0xFF1D4ED8),
                ),
                const SizedBox(width: 4),
                Text(
                  _useLocalModel ? 'On-Device' : 'API',
                  style: GoogleFonts.kanit(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: _useLocalModel ? const Color(0xFF15803D) : const Color(0xFF1D4ED8),
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.settings, color: Color(0xFF1A1A1A)),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SpinKitThreeBounce(
                    color: Color(0xFFFFC93C),
                    size: 40.0,
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'กำลังสกัดค่าสีและทำนายผล...',
                    style: GoogleFonts.kanit(fontSize: 16, color: const Color(0xFF1A1A1A)),
                  ),
                ],
              ),
            )
          : SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 10),
                    // Instructions (3 steps)
                    _buildStepRow('1.', 'เลือกวิธีนำเข้าภาพจากการถ่ายรูปหรือคลังรูปภาพ'),
                    const SizedBox(height: 14),
                    _buildStepRow('2.', 'รอระบบวิเคราะห์'),
                    const SizedBox(height: 14),
                    _buildStepRow('3.', 'เมื่อวิเคราะห์เสร็จจะปรากฏคะแนนสีของไข่แดง'),
                    const Spacer(),
                    // Bottom Buttons
                    SizedBox(
                      width: double.infinity,
                      height: 48,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF1A1A1A),
                          foregroundColor: Colors.white,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(24),
                          ),
                        ),
                        onPressed: () => _pickAndProcessImage(ImageSource.camera),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text('📷 ', style: TextStyle(fontSize: 16)),
                            Text(
                              'ถ่ายรูป',
                              style: GoogleFonts.kanit(fontSize: 15, fontWeight: FontWeight.w600),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      height: 48,
                      child: OutlinedButton(
                        style: OutlinedButton.styleFrom(
                          foregroundColor: const Color(0xFF1A1A1A),
                          side: const BorderSide(color: Color(0xFF1A1A1A), width: 1.5),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(24),
                          ),
                        ),
                        onPressed: () => _pickAndProcessImage(ImageSource.gallery),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text('🖼 ', style: TextStyle(fontSize: 16)),
                            Text(
                              'คลังรูปภาพ',
                              style: GoogleFonts.kanit(fontSize: 15, fontWeight: FontWeight.w600),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildStepRow(String number, String text) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          number,
          style: GoogleFonts.kanit(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            color: const Color(0xFF1A1A1A),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: GoogleFonts.kanit(
              fontSize: 14,
              color: const Color(0xFF333333),
              height: 1.4,
            ),
          ),
        ),
      ],
    );
  }
}
