# local drink finder ☕

**Find your perfect café or boba drink based on what you're craving — using just a zip code and AI-powered matching.**

This app uses either a locally hosted LLM (via Ollama) or Hugging Face's cloud inference API, along with Google Places API, to interpret natural language drink requests and match them to real café and boba shop menus near you.

Built and designed by Amra Ibrahim.

---

## features

- Natural-language drink matching (e.g., "sweet iced oat latte with boba")
- Zip code-based location search
- Two versions available:
  - **Local**: uses Mistral via Ollama (runs offline)
  - **Cloud**: uses Hugging Face's hosted models (no setup needed)
- Google Maps API integration for real café location data
- Streamlit UI for a clean and interactive experience

---

## installation (local version)

1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/local-drink-finder.git
   cd local-drink-finder
