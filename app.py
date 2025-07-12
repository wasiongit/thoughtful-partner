import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import ast
from io import StringIO
from openai import OpenAI
import os

# Config
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX_PATH = "date_ideas.index"
ID_MAP_PATH = "date_ideas_id_map.pkl"

# Initialize OpenAI client
client = OpenAI()


# Utility Functions
def row_to_text(row, row_type="Date Idea"):
    text = f"{row_type}:\n"
    for col, val in row.items():
        if pd.isnull(val):
            continue
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            try:
                val = ", ".join(ast.literal_eval(val))
            except Exception:
                pass
        text += f"{col}: {val}\n"
    return text

def embed_batch(text_list):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text_list
    )
    return np.vstack([np.array(res.embedding, dtype=np.float32) for res in response.data])

def embed_query(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def format_matches_for_context(matches):
    context = ""
    for rank, (score, idea_text) in enumerate(matches, 1):
        context += f"Match {rank} (Score: {score:.2f}):\n{idea_text.strip()}\n\n"
    return context.strip()

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "id_map" not in st.session_state:
    st.session_state.id_map = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "top_n_matches" not in st.session_state:
    st.session_state.top_n_matches = []

# App Layout
st.set_page_config(page_title="Thoughtful Partner", page_icon="")
st.title("Thougtful Date Ideas Assistant")

tab1, tab2, tab3 = st.tabs(["📂 Upload & Embed", "🔍 Search Ideas", "💬 Chat with GPT"])

# =========================
# 📂 Tab 1: Upload & Embed
# =========================
with tab1:
    st.header("Upload Date Ideas CSV & Create Embeddings")
    uploaded_file = st.file_uploader("Upload CSV file with Date Ideas", type="csv")

    if uploaded_file is not None:
        content = StringIO(uploaded_file.getvalue().decode("utf-8"))
        ideas_df = pd.read_csv(content)

        st.write("📄 **CSV Preview:**")
        st.dataframe(ideas_df.head())

        if st.button("Embed and Save Index"):
            with st.spinner("Embedding and indexing date ideas..."):
                idea_texts = ideas_df.apply(lambda row: row_to_text(row, "Date Idea"), axis=1).tolist()
                idea_embeddings = embed_batch(idea_texts)
                faiss.normalize_L2(idea_embeddings)

                dim = idea_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(idea_embeddings)

                # Save index and ID map
                faiss.write_index(index, FAISS_INDEX_PATH)
                id_map = {
                    idx: {"text": idea_texts[idx]} for idx in range(len(idea_texts))
                }
                with open(ID_MAP_PATH, "wb") as f:
                    pickle.dump(id_map, f)

                st.session_state.faiss_index = None
                st.session_state.id_map = None
            st.success("✅ Embeddings created and saved! Please go to Tab 2 and click 'Refresh Embeddings' to load them.")

# =========================
# 🔍 Tab 2: Chat-style QA on Date Ideas
# =========================
# =========================
# 🔍 Tab 2: Chat-style QA on Date Ideas (Improved Layout)
# =========================
with tab2:
    st.header("Ask Anything About Date Ideas")

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
            st.success("✅ Embeddings loaded!")

    if st.session_state.faiss_index is None or st.session_state.id_map is None:
        st.warning("⚠️ Embeddings not loaded yet. Click 'Refresh Embeddings'.")
    else:
        st.subheader("🔧 Customize Assistant Instructions (System Prompt)")

        default_system_prompt = (
            "You are a creative assistant for romantic and fun date ideas. "
            "You will be given a list of date ideas, and the user will ask questions. "
            "Make sure to use all the provided ideas in your response. "
            "You must answer using only the given ideas — do not make up new ones.\n\n"
            "Here are the ideas:\n{context}"
        )

        user_prompt = st.text_area(
            label="Customize how the assistant should behave",
            value=default_system_prompt,
            height=200,
            key="system_prompt_input"
        )

        st.subheader("💬 Ask a Question")

        user_question = st.text_area(
            "Describe your preferences:",
            height=250,
            placeholder='''E.g. Give me date ideas based on below preferences:
                    User Profile:
                    Relationship Status: Dating
                    Love Languages: Acts of Service, Words of Affirmation
                    Dietary Restrictions: None
                    Favourite Cuisines: Italian, Japanese
                    Cooking Preferences: Cook together''',
            key="user_question_input"
        )

        if st.button("🔍 Answer My Question"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving ideas..."):
                    query_vec = embed_query(user_question)
                    D, I = st.session_state.faiss_index.search(query_vec.reshape(1, -1), k=5)
                    matched_ideas = [st.session_state.id_map[idx]["text"] for idx in I[0]]

                    context_block = "\n\n".join([f"Idea {i+1}:\n{txt}" for i, txt in enumerate(matched_ideas)])
                    full_prompt = user_prompt.replace("{context}", context_block)

                with st.spinner("Generating answer..."):
                    chat_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": full_prompt},
                            {"role": "user", "content": user_question}
                        ]
                    )
                    final_answer = chat_response.choices[0].message.content.strip()

                st.markdown("### 🤖 Assistant's Response")
                st.markdown(final_answer)

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

    for msg in st.session_state.chat_history:
        role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg['content']}")
