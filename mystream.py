import streamlit as st
from src import CONFIG
from src.utils import load_collection, create_collection_from_pdf, delete_collection, get_existing_collections
from src.chat import init_llm, display_chat_history, handle_user_input

BASE_DIR = "./collections_chroma"

st.set_page_config(page_title="Assistant SGCI", page_icon="🤖", layout="centered")
st.header("Assistant Personnel SGCI 🤖")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Upload PDF
uploaded_file = st.file_uploader("📂 Chargez un fichier PDF", type="pdf")
if uploaded_file:
    collection_name = create_collection_from_pdf(uploaded_file, BASE_DIR)
    st.session_state.active_collection = collection_name
    st.session_state.vector_store = load_collection(collection_name, BASE_DIR)
    if collection_name not in st.session_state.chat_history:
        st.session_state.chat_history[collection_name] = []

# Sélection / Suppression collection
existing_collections = get_existing_collections(BASE_DIR)
if existing_collections:
    col1, col2 = st.columns([3,1])
    with col1:
        selected_collection = st.selectbox("📚 Sélectionnez une collection :", existing_collections)
        if selected_collection:
            st.session_state.active_collection = selected_collection
            st.session_state.vector_store = load_collection(selected_collection, BASE_DIR)
            if selected_collection not in st.session_state.chat_history:
                st.session_state.chat_history[selected_collection] = []
    with col2:
        if st.button("🗑️ Supprimer la collection sélectionnée"):
            if selected_collection:
                delete_collection(selected_collection, BASE_DIR)
                if selected_collection in st.session_state.chat_history:
                    del st.session_state.chat_history[selected_collection]

#LLM & Chat chain
llm, prompt, chain = init_llm(CONFIG['OPENAI_API_KEY'])

# Affichage historique
if st.session_state.vector_store:
    display_chat_history(st, st.session_state.active_collection, st.session_state.chat_history)

    user_input = st.chat_input("💬 Posez votre question ici...")
    if user_input:
        handle_user_input(
            st=st,
            user_input=user_input,
            active_collection=st.session_state.active_collection,
            chat_history=st.session_state.chat_history,
            vector_store=st.session_state.vector_store,
            chain=chain,
            prompt=prompt
        )

# --- Réinitialisation ---
if st.button("🗑️ Réinitialiser la conversation"):
    if st.session_state.active_collection:
        st.session_state.chat_history[st.session_state.active_collection] = []
    st.success("💡 L'historique de la conversation a été réinitialisé.")
