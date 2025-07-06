import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from io import StringIO
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI()

# Config
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX_PATH = "date_ideas.index"
ID_MAP_PATH = "date_ideas_id_map.pkl"

# Helper Functions
def row_to_text(row, row_type="Date Idea"):
    text = f"{row_type}:\n"
    for col, val in row.items():
        if pd.isnull(val):
            continue
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            try:
                val = ", ".join(eval(val))
            except Exception:
                pass
        text += f"{col}: {val}\n"
    return text

def embed_text(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def embed_query(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "id_map" not in st.session_state:
    st.session_state.id_map = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App Layout
st.set_page_config(page_title="💖 Date Ideas Assistant", page_icon="💖")
st.title("💖 Date Ideas Assistant")

tab1, tab2, tab3 = st.tabs(["📂 Upload & Embed", "🔍 Search Ideas", "💬 Chat with GPT"])

# =========================
# 📂 Tab 1: Upload & Embed
# =========================
with tab1:
    st.header("Upload Date Ideas CSV & Create Embeddings")

    uploaded_file = st.file_uploader("Upload CSV file with Date Ideas", type="csv")

    if uploaded_file is not None:
        # Read CSV
        content = StringIO(uploaded_file.getvalue().decode("utf-8"))
        ideas_df = pd.read_csv(content)

        st.write("📄 **CSV Preview:**")
        st.dataframe(ideas_df.head())

        if st.button("Embed and Save Index"):
            with st.spinner("Embedding and indexing date ideas..."):
                # Create embeddings
                idea_texts = ideas_df.apply(lambda row: row_to_text(row, "Date Idea"), axis=1)
                idea_embeddings = np.vstack([embed_text(t) for t in idea_texts])
                faiss.normalize_L2(idea_embeddings)

                # Build FAISS index
                embedding_dim = idea_embeddings.shape[1]
                index = faiss.IndexFlatIP(embedding_dim)
                index.add(idea_embeddings)

                # Save index to disk
                faiss.write_index(index, FAISS_INDEX_PATH)

                # Save ID map to disk
                id_map = {
                    idx: {
                        "Date Idea": ideas_df.iloc[idx].get("Date Idea", "N/A"),
                        "Category": ideas_df.iloc[idx].get("Category", "N/A")
                    }
                    for idx in range(len(ideas_df))
                }
                with open(ID_MAP_PATH, "wb") as f:
                    pickle.dump(id_map, f)

                # Clear any in-memory index so user must refresh
                st.session_state.faiss_index = None
                st.session_state.id_map = None

            st.success("✅ Embeddings created and saved! Please go to Tab 2 and click 'Refresh Embeddings' to load them.")

# =========================
# 🔍 Tab 2: Search Ideas
# =========================
with tab2:
    st.header("Search for Date Ideas")

    # Refresh Button
    if st.button("🔄 Refresh Embeddings"):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
            st.error("Embedding files not found. Please upload and embed date ideas first.")
        else:
            with st.spinner("Loading embeddings..."):
                index = faiss.read_index(FAISS_INDEX_PATH)
                with open(ID_MAP_PATH, "rb") as f:
                    id_map = pickle.load(f)
                st.session_state.faiss_index = index
                st.session_state.id_map = id_map
            st.success("✅ Embeddings loaded and ready!")

    # Check if embeddings are loaded
    if st.session_state.faiss_index is None or st.session_state.id_map is None:
        st.warning("⚠️ Embeddings not loaded yet. Click 'Refresh Embeddings' to load them.")
    else:
        index = st.session_state.faiss_index
        id_map = st.session_state.id_map

        user_input = st.text_area(
            "Describe your preferences",
            height=200,
            placeholder=(
                "Example:\n"
                "Relationship Status: Dating\n"
                "Love Languages: Quality Time\n"
                "Favourite Cuisines: Italian, Thai\n"
                "Cooking Preferences: Cook together"
            )
        )

        if st.button("Find Date Ideas"):
            if user_input.strip() == "":
                st.warning("Please enter your preferences.")
            else:
                with st.spinner("Searching..."):
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

# =========================
# 💬 Tab 3: Chat with GPT
# =========================
with tab3:
    st.header("Chat with OpenAI Assistant")

    user_message = st.text_input("Your message:")

    if st.button("Send"):
        if user_message.strip() == "":
            st.warning("Please enter a message.")
        else:
            # Append user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_message})

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in relationship and date ideas."},
                        *st.session_state.chat_history
                    ]
                )
                assistant_reply = response.choices[0].message.content.strip()

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **Assistant:** {msg['content']}")
