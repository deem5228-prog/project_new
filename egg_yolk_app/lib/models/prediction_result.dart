class RGBColor {
  final double r;
  final double g;
  final double b;

  RGBColor({required this.r, required this.g, required this.b});

  factory RGBColor.fromJson(Map<String, dynamic> json) {
    return RGBColor(
      r: (json['r'] as num).toDouble(),
      g: (json['g'] as num).toDouble(),
      b: (json['b'] as num).toDouble(),
    );
  }
}

class CIELABColor {
  final double l;
  final double a;
  final double b;
  final double chroma;
  final double hueAngle;

  CIELABColor({
    required this.l,
    required this.a,
    required this.b,
    required this.chroma,
    required this.hueAngle,
  });

  factory CIELABColor.fromJson(Map<String, dynamic> json) {
    return CIELABColor(
      l: (json['l'] as num).toDouble(),
      a: (json['a'] as num).toDouble(),
      b: (json['b'] as num).toDouble(),
      chroma: (json['chroma'] != null) ? (json['chroma'] as num).toDouble() : 0.0,
      hueAngle: (json['hue_angle'] != null) ? (json['hue_angle'] as num).toDouble() : 0.0,
    );
  }
}

class PredictionResult {
  final int predictedScore;
  final double rawScore;
  final RGBColor rgb;
  final CIELABColor cielab;

  PredictionResult({
    required this.predictedScore,
    required this.rawScore,
    required this.rgb,
    required this.cielab,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      predictedScore: json['predicted_score'] as int,
      rawScore: (json['raw_score'] as num).toDouble(),
      rgb: RGBColor.fromJson(json['rgb'] as Map<String, dynamic>),
      cielab: CIELABColor.fromJson(json['cielab'] as Map<String, dynamic>),
    );
  }
}
