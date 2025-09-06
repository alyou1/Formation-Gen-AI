import os
import tempfile
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from src import CONFIG

OPENAI_API_KEY = CONFIG.get('OPENAI_API_KEY')

# Embeddings partagé
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

CHUNK_SIZE = 520
CHUNK_OVERLAP = 20

def load_collection(collection_name: str, base_dir: str) -> Chroma:
    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=os.path.join(base_dir, collection_name)
        )
    except Exception as e:
        raise RuntimeError(f"Impossible de charger la collection {collection_name}: {e}")

def create_collection_from_pdf(uploaded_file, base_dir: str):
    """
    Crée une collection Chroma à partir d'un PDF et affiche une barre de progression.
    """
    import streamlit as st  # nécessaire pour st.progress()

    collection_name = os.path.splitext(uploaded_file.name)[0]
    collection_path = os.path.join(base_dir, collection_name)

    if os.path.exists(collection_path):
        st.info(f"ℹ️ La collection '{collection_name}' existe déjà.")
        return collection_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=520, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=collection_path
    )

    # Barre de progression
    st.info("📄 Indexation en cours...")
    progress_bar = st.progress(0)
    total_docs = len(docs)
    for i, doc in enumerate(docs, start=1):
        vector_store.add_documents(ids=[str(uuid4())], documents=[doc])
        progress_bar.progress(i / total_docs)

    st.success(f"✅ Collection '{collection_name}' créée et indexée !")

    # nettoyer le fichier temporaire
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return collection_name


def delete_collection(collection_name: str, base_dir: str):
    collection_path = os.path.join(base_dir, collection_name)
    if os.path.exists(collection_path):
        for root, dirs, files in os.walk(collection_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(collection_path)

def get_existing_collections(base_dir: str):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
