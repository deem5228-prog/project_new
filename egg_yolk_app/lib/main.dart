import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/splash_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const EggYolkApp());
}

class EggYolkApp extends StatelessWidget {
  const EggYolkApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Egg Yolk Color Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF141421),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFFFB8500),
          secondary: Color(0xFFFFB703),
          surface: Color(0xFF1E1E2C),
        ),
        textTheme: GoogleFonts.kanitTextTheme(
          ThemeData.dark().textTheme,
        ),
      ),
      home: const SplashScreen(),
    );
  }
}
