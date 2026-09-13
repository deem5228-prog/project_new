import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:egg_yolk_app/services/auto_crop_service.dart';
import 'package:egg_yolk_app/services/local_predict_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('Test On-Device AutoCropService on real yolk image', () async {
    final sampleFile = File(r'c:\Users\Admin\Downloads\project_code\pic_egg_yolk\class15\1.jpg');
    if (!sampleFile.existsSync()) {
      print('Sample file not found, skipping: ${sampleFile.path}');
      return;
    }

    final detected = await AutoCropService.detectYolk(sampleFile);
    print('AutoCrop Result: isDetected=${detected.isDetected}, X=${detected.normX}, Y=${detected.normY}, W=${detected.normWidth}, H=${detected.normHeight}');

    expect(detected.isDetected, isTrue);
    expect(detected.normX, greaterThanOrEqualTo(0.0));
    expect(detected.normY, greaterThanOrEqualTo(0.0));
    expect(detected.normWidth, greaterThan(0.0));
    expect(detected.normHeight, greaterThan(0.0));
  });

  test('Test On-Device LocalPredictService on cropped yolk image', () async {
    final sampleCropped = File(r'c:\Users\Admin\Downloads\project_code\egg_yolk_project\model_dev\data\auto_cropped_dataset\class15\1.jpg');
    if (!sampleCropped.existsSync()) {
      print('Cropped sample not found, skipping: ${sampleCropped.path}');
      return;
    }

    final result = await LocalPredictService.predictImage(sampleCropped);
    print('LocalPredict Result: Predicted=${result.predictedScore}, Raw=${result.rawScore}');
    print('RGB: R=${result.rgb.r}, G=${result.rgb.g}, B=${result.rgb.b}');
    print('CIELAB: L=${result.cielab.l}, a=${result.cielab.a}, b=${result.cielab.b}');

    expect(result.predictedScore, inInclusiveRange(4, 15));
    expect(result.rawScore, greaterThan(10.0)); // Class 15 should have high score
    expect(result.rgb.r, greaterThan(100.0));
  });
}
