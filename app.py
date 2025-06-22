import os
import certifi
from dotenv import load_dotenv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import streamlit as st
from geopy.distance import distance as geodistance
import googlemaps
import pgeocode
import random

# Cert fix
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")
HF_API_KEY = os.getenv("hf_api_key")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Broad list of cafe drinks to filter by
CAFE_DRINKS = [
    "espresso", "latte", "cappuccino", "americano", "cold brew", "macchiato", "mocha",
    "flat white", "matcha", "chai latte", "green tea", "black tea", "oolong tea",
    "herbal tea", "bubble tea", "milk tea", "taro milk tea", "thai tea", "iced coffee",
    "drip coffee", "frappuccino", "smoothie", "honeydew smoothie", "mango smoothie",
    "strawberry smoothie", "blueberry smoothie", "pineapple smoothie", "banana smoothie",
    "vanilla latte", "caramel latte", "peach tea", "lemon tea", "hibiscus tea", "acai refresher",
    "refreshers", "boba", "brown sugar boba", "fruit tea", "peppermint mocha",
    "hazelnut latte", "rose latte", "lavender latte", "pumpkin spice latte",
    "white chocolate mocha", "coconut milk tea", "almond milk latte", "dirty chai",
    "iced mocha", "iced espresso", "café au lait", "breve", "nitro cold brew",
    "iced americano", "iced caramel macchiato", "iced matcha", "iced vanilla latte",
    "chai frappuccino", "java chip frappuccino", "iced herbal tea", "london fog",
    "shaken espresso", "flat black", "cortado", "maple latte", "espresso tonic",
    "turmeric latte", "rose matcha", "chocolate cold foam", "toasted vanilla shaken espresso",
    "hazelnut macchiato", "cinnamon dolce latte", "salted caramel cold brew"
]

# Determine if the parsed drink is cafe-appropriate
def is_cafe_drink(query):
    return any(drink in query.lower() for drink in CAFE_DRINKS)

# Convert ZIP to (lat, lon)
def zip_to_coords(zipcode):
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(zipcode)
    if np.isnan(location.latitude) or np.isnan(location.longitude):
        return None
    return (location.latitude, location.longitude)

# Sample drinks per café
def generate_menu():
    return random.sample(CAFE_DRINKS, k=4)

# Google Places → café list
def get_nearby_shops(lat, lon):
    results = gmaps.places_nearby(
        location=(lat, lon),
        radius=5000,
        type="cafe",
        keyword="coffee OR tea OR boba"
    )
    shops = []
    for r in results.get("results", []):
        name = r.get("name")
        loc = r["geometry"]["location"]
        shops.append({
            "shop": name,
            "location": [loc["lat"], loc["lng"]],
            "menu": generate_menu()
        })
    return shops

# Match drink using Hugging Face + embeddings
def match_drink(user_input, user_location, shops):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    payload = {
        "inputs": f"Summarize this drink request as a short, specific cafe-style drink. Return only drinks typically found at cafes like coffee, tea, boba, smoothies, etc. Avoid anything that would not be sold in a cafe: {user_input}",
        "parameters": {"max_new_tokens": 15, "return_full_text": False}
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers=headers,
        json=payload
    )
    result = response.json()
    parsed_query = result[0]['generated_text'].strip() if isinstance(result, list) and result else user_input

    if not is_cafe_drink(parsed_query):
        return parsed_query, []

    input_vector = embedder.encode([parsed_query])
    results = []

    for shop in shops:
        menu_vectors = embedder.encode(shop['menu'])
        similarities = cosine_similarity(input_vector, menu_vectors)
        best_index = np.argmax(similarities)
        best_score = float(similarities[0][best_index])
        best_item = shop['menu'][best_index]
        dist = geodistance(user_location, tuple(shop['location'])).miles

        results.append({
            "shop": shop['shop'],
            "match": best_item,
            "score": round(best_score, 3),
            "distance": round(dist, 2)
        })

    return parsed_query, sorted(results, key=lambda x: (-x['score'], x['distance']))

# Streamlit UI
st.set_page_config(page_title="Local Drink Finder")
st.title("☕ local drink finder")
st.write("find real cafés near your ZIP code for your ideal drink!")

user_input = st.text_input("what kind of drink are you craving today?")
zipcode = st.text_input("enter your ZIP code", max_chars=5)

if user_input and zipcode:
    user_location = zip_to_coords(zipcode)
    if user_location is None:
        st.error("invalid ZIP code. pls try again.")
    else:
        with st.spinner("searching for nearby cafés and matching your drink..."):
            if shops := get_nearby_shops(*user_location):
                parsed, matches = match_drink(user_input, user_location, shops)
                st.subheader(f"interpreted as: *{parsed}*")
                st.markdown("---")
                if not matches:
                    st.warning("no valid cafe drink match found for that request.")
                else:
                    for r in matches:
                        st.markdown(
                            f"**{r['shop']}** — *{r['match']}*  \n"
                            f"similarity score: `{r['score']}` | distance: `{r['distance']} miles`"
                        )
            else:
                st.error("no shops found nearby :( )")

st.markdown("<div style='text-align: right; font-size: small;'>by Amra Ibrahim</div>", unsafe_allow_html=True)
