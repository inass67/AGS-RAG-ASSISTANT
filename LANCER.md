# Lancer le projet (VS Code, terminal PowerShell)
Fichier > Ouvrir le dossier > ags-rag-assistant

    py -3.11 -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env      (puis mets ta clé: GROQ_API_KEY=...)
    python ingest.py
    streamlit run app.py

Connexion par défaut: admin / ags2026
