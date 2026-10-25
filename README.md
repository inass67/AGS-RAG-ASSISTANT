# AGS GROUPE AI Assistant

A Retrieval-Augmented Generation (RAG) assistant built during my PFA internship at AGS Group (Marrakech).
It answers questions from the company's own documents, with sources, instead of relying on a model's memory.

![Home](docs/home.png)

## Features
- Question answering over internal documents (RAG) with cited sources
- Multilingual answers: French, Arabic, Moroccan Darija
- Voice answers (gTTS)
- Construction site photo analysis (BLIP + LLM)
- PDF export of conversations
- Authentication: login and sign-up

![Chat](docs/chat.png)
![Photo analysis](docs/photo.png)

## How it works
Documents → chunking → embeddings (Sentence-Transformers MiniLM) → vector search (ChromaDB, top 4) → grounded generation (Groq LLM via LangChain) → Streamlit UI.

## Tech stack
Python, Streamlit, LangChain, ChromaDB, Sentence-Transformers, Groq, BLIP, gTTS, FPDF2.

## Run locally (Python 3.11)
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # then set GROQ_API_KEY=your_key
python ingest.py           # builds the vector database from data/
streamlit run app.py
```
Default account: `admin` / `ags2026` (change it after the first login).

## Project structure
```
app.py            Streamlit app (UI, auth, RAG chain, image analysis, PDF export)
ingest.py         Indexes the documents in data/ into ChromaDB
data/             Knowledge base (.txt / .pdf)
requirements.txt  Dependencies
```
