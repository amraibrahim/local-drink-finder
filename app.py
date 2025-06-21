import os
import certifi
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from geopy.distance import distance as geodistance
import googlemaps
import pgeocode
import random
import requests

# Set environment for SSL certs
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load secrets
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
GOOGLE_API_KEY = st.secrets["google_api_key"]

# Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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

# Hugging Face fallback-safe summarizer
def summarize_drink_query(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": f"summarize this drink request as a short, specific drink name: {prompt}",
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.7
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"].strip()
    except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
        return prompt.strip()

# Drink match logic
def match_drink(user_input, user_location, shops):
    parsed_query = summarize_drink_query(user_input)
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
                for r in matches:
                    st.markdown(
                        f"**{r['shop']}** — *{r['match']}*  \n"
                        f"Score: `{r['score']}` | distance: `{r['distance']} miles`"
                    )
            else:
                st.error("no shops found nearby :( )")
