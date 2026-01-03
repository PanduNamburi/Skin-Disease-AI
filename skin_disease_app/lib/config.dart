/// API Configuration
/// Update this URL after deploying your Django backend to the cloud
class ApiConfig {
  // For local testing (WiFi only)
  static const String localBaseUrl = 'http://192.168.1.26:8000';
  
  // For cloud deployment (works on mobile data)
  // Replace with your Render/Railway URL after deployment
  // Render example: 'https://skinsense-ai.onrender.com'
  // Railway example: 'https://your-app-name.up.railway.app'
  static const String cloudBaseUrl = 'https://YOUR-APP-NAME.onrender.com';
  
  // Set to true to use cloud URL, false for local
  static const bool useCloud = false;
  
  // Base URL getter
  static String get baseUrl => useCloud ? cloudBaseUrl : localBaseUrl;
  
  // API endpoints
  static String get loginUrl => '$baseUrl/api/login/';
  static String get signupUrl => '$baseUrl/api/signup/';
  static String get predictUrl => '$baseUrl/api/predict/';
}

