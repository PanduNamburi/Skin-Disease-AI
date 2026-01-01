import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'config.dart';

void main() {
  runApp(const SkinDiseaseApp());
}

// --- Localization Config ---
final Map<String, Map<String, String>> _localizedValues = {
  'en': {
    'title': 'SkinSense AI',
    'upload': 'Upload Skin Image',
    'subtitle': 'JPG or PNG preferred',
    'camera': 'Camera',
    'gallery': 'Gallery',
    'analyze': 'ANALYZE IMAGE',
    'disclaimer': '⚠️ Not for clinical use. Always consult a professional.',
    'analysis_complete': 'Analysis Complete',
    'results_ready': 'Your detection results are ready',
    'diagnosis': 'PRIMARY DIAGNOSIS',
    'confidence': 'Confidence Level',
    'severity': 'Severity',
    'predictions': 'Top 3 Predictions',
    'about': 'About',
    'treatment': 'Treatment Recommendations',
    'medical_disclaimer': 'Medical Disclaimer',
    'select_language': 'Select Language',
    'welcome': 'Welcome to SkinSense AI',
    'subtitle_text': 'Advanced AI-powered skin disease detection',
    'get_started': 'Get Started',
    'intro_title': 'Your AI-Powered Skin Health Assistant',
    'intro_subtitle': 'Get instant insights about your skin condition with advanced deep learning technology',
    'feature1_title': 'AI-Powered Detection',
    'feature1_desc': 'Advanced ResNet101 model trained on thousands of skin images for accurate diagnosis',
    'feature2_title': 'Instant Analysis',
    'feature2_desc': 'Get results in seconds with detailed confidence levels and severity assessment',
    'feature3_title': 'Treatment Guidance',
    'feature3_desc': 'Receive personalized treatment recommendations and when to consult a professional',
    'feature4_title': 'Multi-Language Support',
    'feature4_desc': 'Available in English, Hindi, and Telugu for your convenience',
  },
  'hi': {
    'title': 'स्किनसेंस AI',
    'upload': 'त्वचा की छवि अपलोड करें',
    'subtitle': 'JPG या PNG बेहतर है',
    'camera': 'कैमरा',
    'gallery': 'गैलरी',
    'analyze': 'छवि का विश्लेषण करें',
    'disclaimer': '⚠️ नैदानिक उपयोग के लिए नहीं। हमेशा एक पेशेवर से परामर्श करें।',
    'analysis_complete': 'विश्लेषण पूर्ण',
    'results_ready': 'आपके परिणाम तैयार हैं',
    'diagnosis': 'मुख्य निदान',
    'confidence': 'आत्मविश्वास का स्तर',
    'severity': 'गंभीरता',
    'predictions': 'शीर्ष 3 भविष्यवाणियां',
    'about': 'इसके बारे में',
    'treatment': 'उपचार की सिफारिशें',
    'medical_disclaimer': 'चिकित्सा अस्वीकरण',
    'select_language': 'भाषा चुनें',
    'welcome': 'स्किनसेंस AI में आपका स्वागत है',
    'subtitle_text': 'उन्नत AI-संचालित त्वचा रोग पहचान',
    'get_started': 'शुरू करें',
    'intro_title': 'आपका AI-संचालित त्वचा स्वास्थ्य सहायक',
    'intro_subtitle': 'उन्नत डीप लर्निंग तकनीक के साथ अपनी त्वचा की स्थिति के बारे में तत्काल जानकारी प्राप्त करें',
    'feature1_title': 'AI-संचालित पहचान',
    'feature1_desc': 'सटीक निदान के लिए हजारों त्वचा छवियों पर प्रशिक्षित उन्नत ResNet101 मॉडल',
    'feature2_title': 'तत्काल विश्लेषण',
    'feature2_desc': 'विस्तृत आत्मविश्वास स्तर और गंभीरता मूल्यांकन के साथ सेकंडों में परिणाम प्राप्त करें',
    'feature3_title': 'उपचार मार्गदर्शन',
    'feature3_desc': 'व्यक्तिगत उपचार सिफारिशें प्राप्त करें और जानें कि कब एक पेशेवर से परामर्श करना है',
    'feature4_title': 'बहु-भाषा समर्थन',
    'feature4_desc': 'आपकी सुविधा के लिए अंग्रेजी, हिंदी और तेलुगु में उपलब्ध',
  },
  'te': {
    'title': 'స్కిన్ సెన్స్ AI',
    'upload': 'చర్మం చిత్రాన్ని అప్‌లోడ్ చేయండి',
    'subtitle': 'JPG లేదా PNG ప్రాధాన్యత',
    'camera': 'కెమెరా',
    'gallery': 'గ్యాలరీ',
    'analyze': 'చిత్రాన్ని విశ్లేషించండి',
    'disclaimer': '⚠️ క్లినికల్ ఉపయోగం కోసం కాదు. ఎల్లప్పుడూ నిపుణుడిని సంప్రదించండి.',
    'analysis_complete': 'విశ్లేషణ పూర్తయింది',
    'results_ready': 'మీ ఫలితాలు సిద్ధంగా ఉన్నాయి',
    'diagnosis': 'ప్రాథమిక రోగ నిర్ధారణ',
    'confidence': 'నమ్మక స్థాయి',
    'severity': 'తీవ్రత',
    'predictions': 'టాప్ 3 అంచనాలు',
    'about': 'గురించి',
    'treatment': 'చికిత్స సిఫార్సులు',
    'medical_disclaimer': 'వైద్య నిరాకరణ',
    'select_language': 'భాషను ఎంచుకోండి',
    'welcome': 'స్కిన్ సెన్స్ AIకి స్వాగతం',
    'subtitle_text': 'అధునాతన AI-శక్తితో చర్మ వ్యాధి గుర్తింపు',
    'get_started': 'ప్రారంభించండి',
    'intro_title': 'మీ AI-శక్తితో చర్మ ఆరోగ్య సహాయకుడు',
    'intro_subtitle': 'అధునాతన డీప్ లెర్నింగ్ సాంకేతికతతో మీ చర్మ స్థితి గురించి తక్షణ అంతర్దృష్టులను పొందండి',
    'feature1_title': 'AI-శక్తితో గుర్తింపు',
    'feature1_desc': 'ఖచ్చితమైన రోగ నిర్ధారణ కోసం వేలాది చర్మ చిత్రాలపై శిక్షణ పొందిన అధునాతన ResNet101 మోడల్',
    'feature2_title': 'తక్షణ విశ్లేషణ',
    'feature2_desc': 'వివరణాత్మక నమ్మక స్థాయి మరియు తీవ్రత అంచనాతో సెకన్లలో ఫలితాలను పొందండి',
    'feature3_title': 'చికిత్స మార్గదర్శకత్వం',
    'feature3_desc': 'వ్యక్తిగత చికిత్స సిఫార్సులను పొందండి మరియు ఎప్పుడు నిపుణుడిని సంప్రదించాలో తెలుసుకోండి',
    'feature4_title': 'బహుభాషా మద్దతు',
    'feature4_desc': 'మీ సౌకర్యం కోసం ఇంగ్లీష్, హిందీ మరియు తెలుగులో అందుబాటులో ఉంది',
  }
};

String t(String key, String lang) => _localizedValues[lang]![key] ?? key;

// --- Medical Color Scheme ---
class MedicalColors {
  static const Color primaryBlue = Color(0xFF2196F3);
  static const Color lightBlue = Color(0xFFE3F2FD);
  static const Color mediumBlue = Color(0xFFBBDEFB);
  static const Color teal = Color(0xFF009688);
  static const Color lightTeal = Color(0xFF4DB6AC);
  static const Color white = Color(0xFFFFFFFF);
  static const Color lightGray = Color(0xFFF5F5F5);
  static const Color darkGray = Color(0xFF424242);
  static const Color success = Color(0xFF4CAF50);
  static const Color warning = Color(0xFFFF9800);
  static const Color error = Color(0xFFF44336);
}

class SkinDiseaseApp extends StatefulWidget {
  const SkinDiseaseApp({super.key});

  @override
  State<SkinDiseaseApp> createState() => _SkinDiseaseAppState();
}

class _SkinDiseaseAppState extends State<SkinDiseaseApp> {
  String _currentLanguage = 'en';

  @override
  void initState() {
    super.initState();
    _loadLanguage();
  }

  Future<void> _loadLanguage() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _currentLanguage = prefs.getString('language') ?? 'en';
    });
  }

  void setLanguage(String lang) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('language', lang);
    setState(() => _currentLanguage = lang);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SkinSense AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Roboto',
        colorScheme: ColorScheme.fromSeed(
          seedColor: MedicalColors.primaryBlue,
          primary: MedicalColors.primaryBlue,
          secondary: MedicalColors.teal,
          surface: MedicalColors.white,
          background: MedicalColors.lightGray,
        ),
        cardTheme: CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          color: MedicalColors.white,
        ),
      ),
      home: SplashScreen(
        onLanguageSet: setLanguage,
        currentLanguage: _currentLanguage,
      ),
    );
  }
}

// --- 1. Splash Screen ---
class SplashScreen extends StatefulWidget {
  final Function(String) onLanguageSet;
  final String currentLanguage;

  const SplashScreen({super.key, required this.onLanguageSet, required this.currentLanguage});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Timer(const Duration(seconds: 3), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => LanguageSelectionScreen(
          onLanguageSet: widget.onLanguageSet,
          currentLanguage: widget.currentLanguage,
        )),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: MedicalColors.primaryBlue,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [MedicalColors.primaryBlue, MedicalColors.teal],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: MedicalColors.white,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                  ],
                ),
                child: Icon(
                  Icons.health_and_safety_rounded,
                  size: 80,
                  color: MedicalColors.primaryBlue,
                ),
              ),
              const SizedBox(height: 32),
              const Text(
                'SkinSense AI',
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.w800,
                  color: MedicalColors.white,
                  letterSpacing: 1.2,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'Advanced Skin Disease Detection',
                style: TextStyle(
                  fontSize: 16,
                  color: MedicalColors.white,
                  fontWeight: FontWeight.w300,
                ),
              ),
              const SizedBox(height: 40),
              const CircularProgressIndicator(
                color: MedicalColors.white,
                strokeWidth: 3,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// --- 2. Language Selection Screen ---
class LanguageSelectionScreen extends StatelessWidget {
  final Function(String) onLanguageSet;
  final String currentLanguage;

  const LanguageSelectionScreen({super.key, required this.onLanguageSet, required this.currentLanguage});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: MedicalColors.lightGray,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Logo Header
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [MedicalColors.primaryBlue, MedicalColors.teal],
                  ),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: MedicalColors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.health_and_safety_rounded,
                        color: MedicalColors.white,
                        size: 32,
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Expanded(
                      child: Text(
                        'SkinSense AI',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                          color: MedicalColors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      t('select_language', currentLanguage),
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w700,
                        color: MedicalColors.darkGray,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Select Language / भाषा चुनें / భాషను ఎంచుకోండి',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 14,
                        color: MedicalColors.darkGray.withOpacity(0.6),
                      ),
                    ),
                    const SizedBox(height: 40),
                    _LanguageCard(
                      icon: Icons.language,
                      label: 'English',
                      sub: 'English',
                      color: MedicalColors.primaryBlue,
                      onTap: () => _handleSelection(context, 'en'),
                    ),
                    const SizedBox(height: 16),
                    _LanguageCard(
                      icon: Icons.language,
                      label: 'हिंदी',
                      sub: 'Hindi',
                      color: MedicalColors.teal,
                      onTap: () => _handleSelection(context, 'hi'),
                    ),
                    const SizedBox(height: 16),
                    _LanguageCard(
                      icon: Icons.language,
                      label: 'తెలుగు',
                      sub: 'Telugu',
                      color: MedicalColors.lightTeal,
                      onTap: () => _handleSelection(context, 'te'),
                    ),
                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _handleSelection(BuildContext context, String lang) {
    onLanguageSet(lang);
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (context) => IntroductionScreen(lang: lang)),
    );
  }
}

class _LanguageCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final String sub;
  final Color color;
  final VoidCallback onTap;

  const _LanguageCard({
    required this.icon,
    required this.label,
    required this.sub,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 28),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: MedicalColors.darkGray,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      sub,
                      style: TextStyle(
                        fontSize: 14,
                        color: MedicalColors.darkGray.withOpacity(0.6),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.arrow_forward_ios, color: color, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}

// --- 3. Introduction Screen ---
class IntroductionScreen extends StatelessWidget {
  final String lang;
  const IntroductionScreen({super.key, required this.lang});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: MedicalColors.lightGray,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              // Logo Header
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [MedicalColors.primaryBlue, MedicalColors.teal],
                  ),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: MedicalColors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.health_and_safety_rounded,
                        color: MedicalColors.white,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        t('title', lang),
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: MedicalColors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              // Hero Section
              Container(
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      MedicalColors.primaryBlue.withOpacity(0.1),
                      MedicalColors.lightGray,
                    ],
                  ),
                ),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: MedicalColors.white,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: MedicalColors.primaryBlue.withOpacity(0.2),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: Icon(
                        Icons.psychology_rounded,
                        size: 64,
                        color: MedicalColors.primaryBlue,
                      ),
                    ),
                    const SizedBox(height: 24),
                    Text(
                      t('intro_title', lang),
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.w800,
                        color: MedicalColors.darkGray,
                        height: 1.2,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      t('intro_subtitle', lang),
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 16,
                        color: MedicalColors.darkGray.withOpacity(0.7),
                        height: 1.5,
                      ),
                    ),
                  ],
                ),
              ),
              // Features Section
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  children: [
                    _FeatureCard(
                      icon: Icons.auto_awesome_rounded,
                      title: t('feature1_title', lang),
                      description: t('feature1_desc', lang),
                      color: MedicalColors.primaryBlue,
                      iconBackground: MedicalColors.lightBlue,
                    ),
                    const SizedBox(height: 16),
                    _FeatureCard(
                      icon: Icons.speed_rounded,
                      title: t('feature2_title', lang),
                      description: t('feature2_desc', lang),
                      color: MedicalColors.teal,
                      iconBackground: MedicalColors.lightTeal.withOpacity(0.2),
                    ),
                    const SizedBox(height: 16),
                    _FeatureCard(
                      icon: Icons.medical_information_rounded,
                      title: t('feature3_title', lang),
                      description: t('feature3_desc', lang),
                      color: MedicalColors.success,
                      iconBackground: MedicalColors.success.withOpacity(0.1),
                    ),
                    const SizedBox(height: 16),
                    _FeatureCard(
                      icon: Icons.language_rounded,
                      title: t('feature4_title', lang),
                      description: t('feature4_desc', lang),
                      color: MedicalColors.warning,
                      iconBackground: MedicalColors.warning.withOpacity(0.1),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 32),
              // Get Started Button
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: SizedBox(
                  width: double.infinity,
                  height: 60,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(builder: (context) => UploadScreen(lang: lang)),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: MedicalColors.primaryBlue,
                      foregroundColor: MedicalColors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: 4,
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          t('get_started', lang),
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.2,
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Icon(Icons.arrow_forward_rounded, size: 24),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 32),
              // Disclaimer
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Card(
                  elevation: 1,
                  color: MedicalColors.warning.withOpacity(0.1),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(color: MedicalColors.warning.withOpacity(0.3)),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.info_outline, color: MedicalColors.warning, size: 24),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            t('disclaimer', lang),
                            style: TextStyle(
                              fontSize: 12,
                              color: MedicalColors.darkGray,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final Color color;
  final Color iconBackground;

  const _FeatureCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.color,
    required this.iconBackground,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: iconBackground,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 32),
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: color,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    description,
                    style: TextStyle(
                      fontSize: 14,
                      color: MedicalColors.darkGray.withOpacity(0.7),
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// --- 4. Upload Screen (Dashboard Style) ---
class UploadScreen extends StatefulWidget {
  final String lang;
  const UploadScreen({super.key, required this.lang});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  XFile? _selectedImage;
  Uint8List? _imageBytes;
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();
  
  final String _apiUrl = kIsWeb ? 'http://localhost:8000/api/predict/' : ApiConfig.predictUrl;

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source, maxWidth: 1024, maxHeight: 1024, imageQuality: 85);
    if (image != null) {
      final bytes = await image.readAsBytes();
      setState(() { _selectedImage = image; _imageBytes = bytes; });
    }
  }

  Future<void> _analyzeImage() async {
    if (_selectedImage == null || _imageBytes == null) return;
    setState(() => _isLoading = true);
    try {
      var request = http.MultipartRequest('POST', Uri.parse(_apiUrl));
      request.files.add(http.MultipartFile.fromBytes('image', _imageBytes!, filename: 'image.jpg'));
      request.fields['model_type'] = 'dl_resnet';
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      if (response.statusCode == 200) {
        final result = json.decode(response.body);
        if (mounted) {
          Navigator.of(context).push(
            MaterialPageRoute(builder: (context) => ResultScreen(result: result, lang: widget.lang)),
          );
        }
      } else {
        _showError('Server error: ${response.statusCode}');
      }
    } catch (e) {
      _showError('Connection error. Make sure the server is running.');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: MedicalColors.error,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = widget.lang;
    return Scaffold(
      backgroundColor: MedicalColors.lightGray,
      body: SafeArea(
        child: Column(
          children: [
            // Logo Header
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [MedicalColors.primaryBlue, MedicalColors.teal],
                ),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: MedicalColors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(
                      Icons.health_and_safety_rounded,
                      color: MedicalColors.white,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      t('title', l),
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        color: MedicalColors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Welcome Card
                    Card(
                      elevation: 3,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      child: Container(
                        padding: const EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: [MedicalColors.lightBlue, MedicalColors.mediumBlue],
                          ),
                        ),
                        child: Column(
                          children: [
                            Icon(
                              Icons.camera_alt_rounded,
                              size: 48,
                              color: MedicalColors.primaryBlue,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              t('upload', l),
                              style: const TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w700,
                                color: MedicalColors.darkGray,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              t('subtitle', l),
                              style: TextStyle(
                                fontSize: 14,
                                color: MedicalColors.darkGray.withOpacity(0.7),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                    // Image Preview Card
                    Card(
                      elevation: 3,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      child: Container(
                        height: 300,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          color: MedicalColors.lightGray,
                        ),
                        child: _imageBytes != null
                            ? ClipRRect(
                                borderRadius: BorderRadius.circular(16),
                                child: Image.memory(_imageBytes!, fit: BoxFit.cover),
                              )
                            : Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.add_photo_alternate_rounded,
                                    size: 64,
                                    color: MedicalColors.darkGray.withOpacity(0.3),
                                  ),
                                  const SizedBox(height: 16),
                                  Text(
                                    t('upload', l),
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: MedicalColors.darkGray.withOpacity(0.5),
                                    ),
                                  ),
                                ],
                              ),
                      ),
                    ),
                    const SizedBox(height: 24),
                    // Action Buttons Dashboard
                    Row(
                      children: [
                        Expanded(
                          child: _DashboardButton(
                            icon: Icons.camera_alt_rounded,
                            label: t('camera', l),
                            color: MedicalColors.primaryBlue,
                            onPressed: () => _pickImage(ImageSource.camera),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _DashboardButton(
                            icon: Icons.photo_library_rounded,
                            label: t('gallery', l),
                            color: MedicalColors.teal,
                            onPressed: () => _pickImage(ImageSource.gallery),
                          ),
                        ),
                      ],
                    ),
                    if (_selectedImage != null) ...[
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        height: 56,
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _analyzeImage,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: MedicalColors.primaryBlue,
                            foregroundColor: MedicalColors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            elevation: 3,
                          ),
                          child: _isLoading
                              ? const SizedBox(
                                  height: 24,
                                  width: 24,
                                  child: CircularProgressIndicator(
                                    color: MedicalColors.white,
                                    strokeWidth: 2,
                                  ),
                                )
                              : Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const Icon(Icons.analytics_rounded, size: 24),
                                    const SizedBox(width: 12),
                                    Text(
                                      t('analyze', l),
                                      style: const TextStyle(
                                        fontSize: 18,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ],
                                ),
                        ),
                      ),
                    ],
                    const SizedBox(height: 24),
                    // Disclaimer Card
                    Card(
                      elevation: 2,
                      color: MedicalColors.warning.withOpacity(0.1),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                        side: BorderSide(color: MedicalColors.warning.withOpacity(0.3)),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Row(
                          children: [
                            Icon(Icons.info_outline, color: MedicalColors.warning, size: 24),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                t('disclaimer', l),
                                style: TextStyle(
                                  fontSize: 12,
                                  color: MedicalColors.darkGray,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DashboardButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onPressed;

  const _DashboardButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 20),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(icon, color: color, size: 28),
              ),
              const SizedBox(height: 12),
              Text(
                label,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: MedicalColors.darkGray,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// --- 5. Result Screen (Dashboard Style) ---
class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;
  final String lang;
  const ResultScreen({super.key, required this.result, required this.lang});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: MedicalColors.lightGray,
      body: SafeArea(
        child: Column(
          children: [
            // Logo Header
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [MedicalColors.primaryBlue, MedicalColors.teal],
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: MedicalColors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: MedicalColors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(
                      Icons.health_and_safety_rounded,
                      color: MedicalColors.white,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      t('title', lang),
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        color: MedicalColors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
        child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
                    // Success Header Card
                    Card(
                      elevation: 3,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                      child: Container(
                        padding: const EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: [MedicalColors.success.withOpacity(0.1), MedicalColors.teal.withOpacity(0.1)],
                          ),
                        ),
                        child: Column(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: MedicalColors.success.withOpacity(0.2),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.check_circle_rounded,
                                size: 48,
                                color: MedicalColors.success,
                              ),
                            ),
                            const SizedBox(height: 16),
            Text(
                              t('analysis_complete', lang),
                              style: const TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.w700,
                                color: MedicalColors.darkGray,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              t('results_ready', lang),
                              style: TextStyle(
                                fontSize: 14,
                                color: MedicalColors.darkGray.withOpacity(0.7),
                              ),
            ),
          ],
        ),
      ),
                    ),
                    const SizedBox(height: 24),
                    // Primary Diagnosis Card
                    _buildDiagnosisCard(result, lang),
                    const SizedBox(height: 20),
                    // Stats Dashboard
                    Row(
                      children: [
                        Expanded(
                          child: _buildStatCard(
                            icon: Icons.analytics_rounded,
                            label: t('confidence', lang),
                            value: '${((result['confidence'] ?? 0.0) * 100).toStringAsFixed(1)}%',
                            color: MedicalColors.primaryBlue,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _buildStatCard(
                            icon: Icons.warning_rounded,
                            label: t('severity', lang),
                            value: result['severity'] ?? 'Unknown',
                            color: _getSeverityColor(result['severity'] ?? 'Unknown'),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 20),
                    // Top Predictions Card
                    _buildPredictionsCard(result, lang),
                    if (result['disease_info'] != null && result['disease_info'].isNotEmpty) ...[
                      const SizedBox(height: 20),
                      _buildTreatmentCard(result, lang),
                    ],
                    const SizedBox(height: 20),
                    // Back Button
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: ElevatedButton(
                        onPressed: () => Navigator.of(context).pop(),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: MedicalColors.primaryBlue,
                          foregroundColor: MedicalColors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          elevation: 3,
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.arrow_back_rounded),
                            const SizedBox(width: 8),
                            const Text(
                              'Back to Upload',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDiagnosisCard(Map<String, dynamic> result, String lang) {
    final diseaseName = result['disease_name']?.toString().replaceAll('_', ' ') ?? 'Unknown';
    final isNormalSkin = result['is_normal_skin'] ?? false;
    final isUnreliable = result['is_unreliable'] ?? false;

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              MedicalColors.primaryBlue.withOpacity(0.1),
              MedicalColors.teal.withOpacity(0.1),
            ],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: MedicalColors.primaryBlue.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.medical_services_rounded,
                    color: MedicalColors.primaryBlue,
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Text(
                    t('diagnosis', lang),
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: MedicalColors.darkGray.withOpacity(0.6),
                      letterSpacing: 1.2,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Text(
              isNormalSkin ? '✅ $diseaseName' : diseaseName,
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                color: MedicalColors.darkGray,
              ),
            ),
            const SizedBox(height: 16),
            if (isUnreliable)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: MedicalColors.warning.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: MedicalColors.warning.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, color: MedicalColors.warning, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        t('medical_disclaimer', lang),
                        style: TextStyle(
                          fontSize: 12,
                          color: MedicalColors.darkGray,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 24),
            ),
            const SizedBox(height: 12),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                color: MedicalColors.darkGray.withOpacity(0.6),
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionsCard(Map<String, dynamic> result, String lang) {
    final topPredictions = result['top_predictions'] as List? ?? [];

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.bar_chart_rounded, color: MedicalColors.primaryBlue, size: 24),
                const SizedBox(width: 12),
                Text(
                  t('predictions', lang),
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: MedicalColors.darkGray,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            ...List.generate(
              topPredictions.length.clamp(0, 3),
              (index) {
                final pred = topPredictions[index];
                final name = pred['disease']?.toString().replaceAll('_', ' ') ?? '';
                final confidence = (pred['confidence'] ?? 0.0) as double;
                return Padding(
                  padding: EdgeInsets.only(bottom: index < 2 ? 16 : 0),
                  child: _buildPredictionItem(index + 1, name, confidence),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionItem(int rank, String name, double confidence) {
    final colors = [
      MedicalColors.primaryBlue,
      MedicalColors.teal,
      MedicalColors.lightTeal,
    ];
    final color = colors[(rank - 1) % colors.length];

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                '$rank',
                style: const TextStyle(
                  color: MedicalColors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              name,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: MedicalColors.darkGray,
              ),
            ),
          ),
          Text(
            '${(confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTreatmentCard(Map<String, dynamic> result, String lang) {
    final diseaseInfo = result['disease_info'] ?? {};
    final diseaseName = result['disease_name']?.toString().replaceAll('_', ' ') ?? '';

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.medical_information_rounded, color: MedicalColors.teal, size: 24),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    '${t('about', lang)} $diseaseName',
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: MedicalColors.darkGray,
                    ),
                  ),
                ),
              ],
            ),
            if (diseaseInfo['description'] != null) ...[
              const SizedBox(height: 16),
              Text(
                diseaseInfo['description'] ?? '',
                style: TextStyle(
                  fontSize: 14,
                  color: MedicalColors.darkGray.withOpacity(0.8),
                  height: 1.6,
                ),
              ),
            ],
            if (diseaseInfo['treatment_recommendations'] != null) ...[
              const SizedBox(height: 24),
              Row(
                children: [
                  Icon(Icons.lightbulb_rounded, color: MedicalColors.warning, size: 24),
                  const SizedBox(width: 12),
                  Text(
                    t('treatment', lang),
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: MedicalColors.darkGray,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ...(diseaseInfo['treatment_recommendations'] as List? ?? []).map((t) => Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.check_circle_rounded,
                          color: MedicalColors.success,
                          size: 20,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            t.toString(),
                            style: TextStyle(
                              fontSize: 14,
                              color: MedicalColors.darkGray.withOpacity(0.8),
                              height: 1.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  )),
            ],
          ],
        ),
      ),
    );
  }

  Color _getSeverityColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'high':
        return MedicalColors.error;
      case 'medium':
        return MedicalColors.warning;
      default:
        return MedicalColors.success;
    }
  }
}
