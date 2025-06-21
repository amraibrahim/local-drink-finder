# local drink finder ☕

**find your perfect café or boba drink based on what you're craving — using just a zip code and a local ai model.**

this app uses a locally hosted llm (via ollama) and google places api to interpret natural language drink requests and match them to real café and boba shop menus near you.

---

## features

- natural-language drink matching (e.g., "sweet iced oat latte with boba")
- zip code-based location search
- local llm (mistral via ollama) — works offline for ai processing
- google maps api integration for real café data
- streamlit ui for a smooth experience

---

## tech stack used for this

- python
- [streamlit](https://streamlit.io)
- [ollama](https://ollama.com) + mistral llm
- [sentence-transformers](https://www.sbert.net)
- [google maps api](https://developers.google.com/maps)

---

## installation

1. **clone the repository**

```bash
git clone https://github.com/amraibrahim/local-drink-finder.git
cd local-drink-finder

