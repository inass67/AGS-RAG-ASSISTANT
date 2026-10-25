"""
Script d'ingestion : lit les documents du dossier data/, les découpe en morceaux
(chunks) et les stocke dans une base vectorielle Chroma locale.

A lancer UNE FOIS (ou à chaque fois que tu ajoutes/modifies un document dans data/) :
    python ingest.py
"""

import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


def load_documents():
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, filename)
        if filename.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif filename.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs


def main():
    print("Chargement des documents depuis data/ ...")
    documents = load_documents()
    print(f"-> {len(documents)} document(s) chargé(s).")

    if not documents:
        print("Aucun document trouvé dans data/. Ajoute des fichiers .txt ou .pdf puis relance.")
        return

    print("Découpage en chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"-> {len(chunks)} chunk(s) créé(s).")

    print("Création des embeddings (téléchargement du modèle la 1ère fois, ~1-2 min)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(PERSIST_DIR):
        print("Ancienne base vectorielle trouvée, suppression...")
        shutil.rmtree(PERSIST_DIR)

    print("Sauvegarde dans la base vectorielle Chroma...")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Base vectorielle prête dans ./{PERSIST_DIR} — tu peux maintenant lancer l'app avec: streamlit run app.py")


if __name__ == "__main__":
    main()
