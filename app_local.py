import os
import certifi
from dotenv import load_dotenv
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
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
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Drink filter to exclude non-drinks or inappropriate inputs
UNWANTED_KEYWORDS = ["alcohol", "vodka", "rum", "tequila", "whiskey", "beer", "wine"]

def is_valid_drink(query):
    lowered = query.lower()
    return not any(keyword in lowered for keyword in UNWANTED_KEYWORDS)

# Convert ZIP to (lat, lon)
def zip_to_coords(zipcode):
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(zipcode)
    if np.isnan(location.latitude) or np.isnan(location.longitude):
        return None
    return (location.latitude, location.longitude)

# Sample drinks per café
def generate_menu():
    base_drinks = [
        "Iced Matcha Latte", "Taro Milk Tea", "Brown Sugar Boba", "Espresso", "Cold Brew",
        "Mango Green Tea", "Thai Tea", "Honeydew Smoothie", "Vanilla Latte", "Peach Refresher"
    ]
    return random.sample(base_drinks, k=4)

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

# Match drink using Ollama + embeddings
def match_drink(user_input, user_location, shops):
    response = ollama.chat(model='mistral', messages=[
        {
            "role": "user",
            "content": f"summarize this drink request as a short, specific drink name or description. it can be coffee, tea, boba, smoothie, etc: {user_input}"
        }
    ])
    parsed_query = response['message']['content'].strip()
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
    if not is_valid_drink(user_input):
        st.warning("Please enter a non-alcoholic drink.")
    else:
        user_location = zip_to_coords(zipcode)
        if user_location is None:
            st.error("invalid ZIP code. pls try again.")
        else:
            with st.spinner("searching for nearby cafés and matching your drink..."):
                if shops := get_nearby_shops(*user_location):
                    parsed, matches = match_drink(user_input, user_location, shops)
                    st.subheader(f"interpreted as: *{parsed}*")
                    st.markdown("---")
                    for r in matches:
                        st.markdown(
                            f"**{r['shop']}** — *{r['match']}*  \n"
                            f"Score: `{r['score']}` | distance: `{r['distance']} miles`"
                        )
                else:
                    st.error("no shops found nearby :( )")

st.markdown("<div style='text-align: right; font-size: small;'>by Amra Ibrahim</div>", unsafe_allow_html=True)