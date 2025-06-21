import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pgeocode
import googlemaps
import requests

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load shop data
shops = pd.read_csv("cafes.csv")

# Geolocation setup
geolocator = pgeocode.Nominatim('us')
gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_API_KEY"])

# Title and UI
st.title("☕ local drink finder")
st.caption("find real cafés near your ZIP code for your ideal drink!")

user_input = st.text_input("what kind of drink are you craving today?")
user_zip = st.text_input("enter your ZIP code")

# Hugging Face query function
def query_huggingface_model(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACEHUB_API_TOKEN']}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 100
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Matching function
def match_drink(drink, location, shop_data):
    prompt = f"You are helping someone find a drink like '{drink}'. List possible matching drinks based on keywords only."
    response_data = query_huggingface_model(prompt)

    try:
        parsed = response_data[0]["generated_text"]
    except (KeyError, IndexError):
        parsed = drink  # fallback

    drink_vector = model.encode([parsed])
    menu_vectors = model.encode(shop_data['menu'].tolist())
    sims = cosine_similarity(drink_vector, menu_vectors)[0]

    shop_data['similarity'] = sims

    nearby = geolocator.query_postal_code(location)
    if nearby is None or pd.isna(nearby.latitude) or pd.isna(nearby.longitude):
        st.error("Invalid ZIP code.")
        return None, None

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    shop_data['distance'] = shop_data.apply(
        lambda row: haversine(nearby.latitude, nearby.longitude, row['latitude'], row['longitude']), axis=1
    )

    filtered = shop_data[shop_data['distance'] <= 20]
    top_matches = filtered.sort_values(by='similarity', ascending=False).head(3)

    return parsed, top_matches

# Run when button is clicked
if st.button("Find drinks") and user_input and user_zip:
    with st.spinner("brewing results..."):
        parsed, matches = match_drink(user_input, user_zip, shops)
        if matches is not None:
            st.subheader("here’s what we found 🍵")
            st.markdown(f"Interpreted your drink as: `{parsed.strip()}`")
            for i, row in matches.iterrows():
                st.markdown(f"**{row['shop']}** — {row['menu']}")
                st.caption(f"{round(row['distance'], 1)} km away")