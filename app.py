import streamlit as st
import faiss
import pickle
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX_PATH = "date_ideas.index"
ID_MAP_PATH = "date_ideas_id_map.pkl"

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load ID map
with open(ID_MAP_PATH, "rb") as f:
    id_map = pickle.load(f)

# Function to embed query text
def embed_query(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# Streamlit App
st.set_page_config(page_title="Date Ideas Matcher", page_icon="💖")
st.title("Thoughtful Partner Algo")

st.write("Enter details about yourself or your preferences to discover tailored date ideas.")

# Multiline text input
user_input = st.text_area(
    "Your Profile & Preferences",
    height=200,
    placeholder=(
        "Example:\n"
        "Relationship Status: Dating\n"
        "Love Languages: Acts of Service, Words of Affirmation\n"
        "Dietary Restrictions: None\n"
        "Favourite Cuisines: Italian, Japanese\n"
        "Cooking Preferences: Cook together"
    )
)

if st.button("Find Date Ideas"):
    if user_input.strip() == "":
        st.warning("Please enter your profile information.")
    else:
        with st.spinner("Finding the best date ideas..."):
            # Embed and search
            query_vec = embed_query(user_input)
            D, I = index.search(query_vec.reshape(1, -1), k=5)

        st.subheader("🎯 Top Matches")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
            meta = id_map[idx]
            st.markdown(
                f"""
                **{rank}. {meta['Date Idea']}**
                
                *Category:* {meta['Category']}  
                *Similarity Score:* {score:.3f}
                """
            )
