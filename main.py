import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# load menu
with open('menus.json') as f:
    menus = json.load(f)
    
# load model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

#function to match drinks
def match_drink(user_input):
    # Ask the local LLM (Mistral) to summarize the drink request
    response = ollama.chat(model='mistral', messages=[
        {
            "role": "user",
            "content": f"Summarize this drink request as a short, specific drink name or description. It can be coffee, tea, boba, smoothie, etc: {user_input}"
        }
    ])
    parsed_query = response['message']['content'].strip()
    print(f"LLM interpreted your drink as: {parsed_query}")
    
    input_vector = embedder.encode([parsed_query])
    results = []

    for shop in menus:
        menu_items = shop['menu']
        menu_vectors = embedder.encode(shop['menu'])
        similarities = cosine_similarity(input_vector, menu_vectors)
        
        best_index = np.argmax(similarities)
        best_score = float(similarities[0][best_index])
        best_item = menu_items[best_index]

        results.append({
            "shop": shop['shop'],
            "match": best_item,
            "score": round(best_score, 3)
        })

    return sorted(results, key=lambda x: -x['score'])

# Run in terminal
if __name__ == "__main__":
    print("welcome to the Local Drink Finder!")
    user_query = input("what kind of drink are you in the mood for?\n> ")
    results = match_drink(user_query)

    print("\nTop Matches:")
    for r in results:
        print(f"- {r['shop']}: {r['match']} (Similarity Score: {r['score']})")