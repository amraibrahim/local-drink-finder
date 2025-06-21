# local drink finder ☕

**find your perfect café or boba drink based on what you're craving — using just a zip code and AI-powered matching.**

this app uses either a locally hosted LLM (via Ollama) or Hugging Face's cloud inference API, along with Google Places API, to interpret natural language drink requests and match them to real café and boba shop menus near you.

built and designed by Amra Ibrahim.

---

# features

- natural-language drink matching (e.g., "sweet iced oat latte with boba")
- zip code-based location search
- two available versions:
  - **cloud version** (no setup needed): uses Hugging Face API
  - **local version** (offline): uses Mistral via Ollama + local vector embeddings
- real café location data via Google Maps API
- streamlit UI for a clean and interactive experience

---

## 🌐 how to access

### cloud version (public):
- available at: **https://yourlocaldrinkfinder.streamlit.app**
- no installation or setup required.

### local version (developer setup):
- run locally on your machine using Python, Ollama, and your own API keys

---

## 🧠 tech stack

- python
- [streamlit](https://streamlit.io)
- [ollama + mistral](https://ollama.com)
- [huggingface](https://huggingface.co)
- [sentence-transformers](https://www.sbert.net)
- [google maps API](https://developers.google.com/maps)
- [scikit-learn](https://scikit-learn.org)
- [pgeocode](https://pypi.org/project/pgeocode)
- [geopy](https://pypi.org/project/geopy)

---

## 🛠️ installation (for local version)

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/local-drink-finder.git
cd local-drink-finder
