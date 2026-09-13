import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_result.dart';

class DetailScreen extends StatelessWidget {
  final PredictionResult result;

  const DetailScreen({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Color(0xFF1A1A1A), size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          'Detail',
          style: GoogleFonts.kanit(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: const Color(0xFF1A1A1A),
          ),
        ),
        centerTitle: false,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildSectionTitle('ค่าสี RGB'),
              const SizedBox(height: 4),
              Text(
                'R : ${result.rgb.r.round()}    G : ${result.rgb.g.round()}    B : ${result.rgb.b.round()}',
                style: GoogleFonts.outfit(fontSize: 14, color: const Color(0xFF444444)),
              ),
              const SizedBox(height: 20),

              _buildSectionTitle('ค่าสี Lab'),
              const SizedBox(height: 4),
              Text(
                'L* : ${result.cielab.l.round()}    a* : ${result.cielab.a.round()}    b* : ${result.cielab.b.round()}',
                style: GoogleFonts.outfit(fontSize: 14, color: const Color(0xFF444444)),
              ),
              const SizedBox(height: 20),

              _buildSectionTitle('ค่า Chroma (ความสดของสี C*)'),
              const SizedBox(height: 4),
              Text(
                result.cielab.chroma.toStringAsFixed(2),
                style: GoogleFonts.outfit(fontSize: 14, color: const Color(0xFF444444)),
              ),
              const SizedBox(height: 20),

              _buildSectionTitle('ค่า Hue angle (มุมของโทนสี h°)'),
              const SizedBox(height: 4),
              Text(
                '${result.cielab.hueAngle.toStringAsFixed(2)}°',
                style: GoogleFonts.outfit(fontSize: 14, color: const Color(0xFF444444)),
              ),
              const SizedBox(height: 20),

              _buildSectionTitle('ค่าคะแนนสีของไข่'),
              const SizedBox(height: 4),
              Text(
                '${result.predictedScore}  (Raw Score: ${result.rawScore.toStringAsFixed(2)})',
                style: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.bold, color: const Color(0xFF1A1A1A)),
              ),
              const SizedBox(height: 20),

              _buildSectionTitle('รายละเอียด'),
              const SizedBox(height: 4),
              Text(
                'คะแนนระดับนี้บ่งบอกความเข้มของสีไข่แดง เทียบกับมาตรฐานพัดสี (Yolk Color Fan 1-15) โดยสกัดค่าสีจากภาพถ่ายและประมวลผลด้วยโมเดล Machine Learning (SVR RBF Kernel)',
                style: GoogleFonts.kanit(fontSize: 13, color: const Color(0xFF666666), height: 1.5),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: GoogleFonts.kanit(
        fontSize: 15,
        fontWeight: FontWeight.bold,
        color: const Color(0xFF1A1A1A),
      ),
    );
  }
}
