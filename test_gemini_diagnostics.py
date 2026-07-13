import sys
import os

# Ensure settings can be loaded
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import GEMINI_API_KEY, GEMINI_MODEL_ID

print("=" * 60)
print("🧠 Gemini API Client Diagnostic Check")
print("=" * 60)

# 1. Check if google-genai is installed
has_new_sdk = False
try:
    from google import genai
    print("✅ New 'google-genai' package is installed.")
    has_new_sdk = True
except ImportError:
    print("❌ New 'google-genai' package is NOT installed.")

# 2. Check if legacy google-generativeai is installed
has_old_sdk = False
try:
    import google.generativeai as old_genai
    print("✅ Legacy 'google-generativeai' package is installed.")
    has_old_sdk = True
except ImportError:
    print("❌ Legacy 'google-generativeai' package is NOT installed.")

# 3. Mask and verify API key
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY is not defined in settings or .env!")
    sys.exit(1)

masked_key = f"{GEMINI_API_KEY[:6]}...{GEMINI_API_KEY[-4:]}" if len(GEMINI_API_KEY) > 10 else "Too Short"
print(f"🔑 Loaded API Key: {masked_key}")

# 4. Test connection using New SDK
if has_new_sdk:
    print("\n--- Testing Connection via NEW google-genai SDK (v1 endpoint) ---")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1"})
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents="Say 'Gemini API is working successfully!'"
        )
        print("🎉 SUCCESS!")
        print("Gemini Response:", response.text)
        sys.exit(0)
    except Exception as new_exc:
        print("❌ New SDK call failed:", new_exc)

# 5. Test connection using Legacy SDK
if has_old_sdk:
    print("\n--- Testing Connection via LEGACY google-generativeai SDK (v1 endpoint override) ---")
    try:
        old_genai.configure(api_key=GEMINI_API_KEY, client_options={"api_version": "v1"})
        model = old_genai.GenerativeModel(GEMINI_MODEL_ID)
        response = model.generate_content("Say 'Gemini API is working successfully!'")
        print("🎉 SUCCESS!")
        print("Gemini Response:", response.text)
        sys.exit(0)
    except Exception as old_exc:
        print("❌ Legacy SDK call failed:", old_exc)

print("\n❌ All connection checks failed. Please make sure you have run 'pip install google-genai' and verified your billing in Google AI Studio.")
sys.exit(1)
