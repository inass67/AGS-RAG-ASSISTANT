"""
Assistant IA - AGS GROUPE
Chatbot RAG (Retrieval-Augmented Generation) interne - version multilingue avec analyse d'images.

Lancement :
    streamlit run app.py
"""

import os
import io
import re
import json
import hashlib
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from gtts import gTTS
from fpdf import FPDF
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PERSIST_DIR = "chroma_db"
LOGO_PATH = "logo_ags.png"
USERS_FILE = "users.json"

st.set_page_config(page_title="Assistant IA - AGS GROUPE", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(900px 500px at 90% -10%, rgba(225,29,72,0.20), transparent 60%),
        radial-gradient(800px 500px at -10% 5%, rgba(59,130,246,0.16), transparent 60%),
        radial-gradient(700px 500px at 50% 120%, rgba(124,58,237,0.12), transparent 60%),
        #0A0F1C;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"] { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1324 0%, #0E1A30 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #D5DCEA !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }
[data-testid="stSidebar"] h2 { font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; color:#FB7185 !important; }
[data-testid="stSidebar"] img { background:#fff; border-radius:16px; padding:10px; max-height:130px; object-fit:contain; }
.user-chip {
    display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:16px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
}
.user-avatar {
    width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#E11D48,#F97316); font-weight:800; color:white !important;
    box-shadow: 0 6px 18px rgba(225,29,72,0.45);
}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.05); padding: 6px; border-radius: 16px; width: fit-content;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 12px; padding: 9px 22px; font-weight: 600; color:#93A0BC; height: auto; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#E11D48,#F43F5E) !important; color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(225,29,72,0.40);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* Boutons */
div.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #E11D48 0%, #F43F5E 100%);
    color: white; border: none; border-radius: 12px; font-weight: 600; padding: 0.6rem 1.3rem;
    box-shadow: 0 8px 22px rgba(225,29,72,0.35); transition: all .18s ease;
}
div.stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); color: white; box-shadow: 0 12px 28px rgba(225,29,72,0.50);
}

/* Cartes de suggestions */
.sugg-title { font-weight:800; color:#FFFFFF; font-size:1.35rem; margin: 26px 0 4px 0; letter-spacing:-0.01em; }
.sugg-sub { color:#8D9AB5; font-size:.95rem; margin-bottom: 16px; }
div[class*="st-key-sugg_"] button {
    background: rgba(255,255,255,0.05) !important; color:#E5E9F2 !important; border:1px solid rgba(255,255,255,0.12) !important;
    box-shadow: none !important; justify-content:flex-start; text-align:left; width:100%; min-height:78px; border-radius:18px; font-weight:500;
}
div[class*="st-key-sugg_"] button:hover {
    border-color:#F43F5E !important; background: rgba(244,63,94,0.12) !important; color:#FFFFFF !important;
    box-shadow: 0 12px 30px rgba(225,29,72,0.25) !important; transform: translateY(-2px);
}

/* Messages de chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:20px;
    padding:16px 20px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(225,29,72,0.28), rgba(124,58,237,0.20)); border: 1px solid rgba(244,63,94,0.30);
}
[data-testid="stChatInput"] { border-radius: 18px; border:1px solid rgba(255,255,255,0.14); box-shadow: 0 10px 40px rgba(0,0,0,0.40); }
[data-testid="stChatInput"]:focus-within { border-color:#F43F5E; box-shadow: 0 0 0 3px rgba(244,63,94,0.20), 0 10px 40px rgba(0,0,0,0.40); }

/* Formulaires / login */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10); border-radius:22px; padding: 28px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.45); backdrop-filter: blur(14px);
}
.stTextInput input { border-radius: 12px; }
[data-testid="stFileUploader"] section { border-radius: 18px; border: 2px dashed rgba(255,255,255,0.20); background: rgba(255,255,255,0.03); }
[data-testid="stExpander"] { border-radius: 16px; border:1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03); }

/* Hero login */
.hero-card { text-align:center; padding: 10px 0 22px 0; }
.logo-chip { display:inline-block; background:#FFFFFF; border-radius:20px; padding:10px 18px; box-shadow: 0 14px 40px rgba(225,29,72,0.35); }
.logo-chip img { height:64px; display:block; }
.hero-badge {
    display:inline-block; margin-top:20px; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#FDA4AF; background: rgba(244,63,94,0.14); border:1px solid rgba(251,113,133,0.35); padding:5px 14px; border-radius:999px;
}
.hero-card h1 { font-weight:800; font-size:2.3rem; color:#FFFFFF; margin: 14px 0 8px 0; letter-spacing:-0.03em; line-height:1.1; }
.hero-card h1 span { background: linear-gradient(90deg,#FB7185,#F97316); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-card p { color:#8D9AB5; margin:0; font-size:1rem; }
.hero-pills { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:16px; }
.hero-pills span { font-size:.78rem; font-weight:600; color:#D5DCEA; padding:5px 12px; border-radius:999px; background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); }
.info-card {
    background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10); border-left:4px solid #F43F5E; border-radius:16px;
    padding:16px 20px; color:#C5CEE0; margin-bottom:16px;
}
.info-card b { color:#FFFFFF; }
</style>
""", unsafe_allow_html=True)


# ---------- GESTION DES COMPTES (fichier local users.json) ----------

def hash_password(password):
    """Hache le mot de passe avec SHA-256 (jamais de mot de passe en clair stocké)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {"admin": hash_password("ags2026")}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f)
        return default_users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f)


def check_login():
    """Affiche les écrans de connexion / inscription. Retourne True si authentifié."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        _logo_html = ""
        if os.path.exists(LOGO_PATH):
            import base64
            with open(LOGO_PATH, "rb") as _f:
                _logo_html = ('<div class="logo-chip"><img src="data:image/png;base64,'
                              + base64.b64encode(_f.read()).decode() + '"></div>')
        st.markdown(
            '<div class="hero-card">' + _logo_html +
            '<div><span class="hero-badge">IA générative · RAG</span></div>'
            '<h1>Assistant IA <span>AGS GROUPE</span></h1>'
            '<p>Posez vos questions sur nos projets, devis et services.</p>'
            '<div class="hero-pills"><span>📚 Base documentaire</span><span>🌐 FR · AR · Darija</span>'
            '<span>🔊 Voix</span><span>📷 Analyse chantier</span></div></div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["🔑 Connexion", "✍️ Créer un compte"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur", key="login_user")
                password = st.text_input("Mot de passe", type="password", key="login_pass")
                submitted = st.form_submit_button("Se connecter")

                if submitted:
                    users = load_users()
                    if username in users and users[username] == hash_password(password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.rerun()
                    else:
                        st.error("Nom d'utilisateur ou mot de passe incorrect.")

        with tab_signup:
            with st.form("signup_form"):
                new_username = st.text_input("Choisissez un nom d'utilisateur", key="signup_user")
                new_password = st.text_input("Choisissez un mot de passe", type="password", key="signup_pass")
                confirm_password = st.text_input("Confirmez le mot de passe", type="password", key="signup_confirm")
                signup_submitted = st.form_submit_button("Créer mon compte")

                if signup_submitted:
                    users = load_users()
                    if not new_username.strip() or not new_password.strip():
                        st.error("Merci de remplir tous les champs.")
                    elif new_username in users:
                        st.error("Ce nom d'utilisateur existe déjà. Choisissez-en un autre.")
                    elif new_password != confirm_password:
                        st.error("Les mots de passe ne correspondent pas.")
                    elif len(new_password) < 4:
                        st.error("Le mot de passe doit contenir au moins 4 caractères.")
                    else:
                        users[new_username] = hash_password(new_password)
                        save_users(users)
                        st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter dans l'onglet Connexion.")

    return False


if not check_login():
    st.stop()

# ---------- A PARTIR D'ICI, TOUT LE RESTE NE S'AFFICHE QUE SI CONNECTE ----------

ROBOT_SVG = """
<svg class="robot-float" width="170" height="150" viewBox="0 0 260 225" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="eyeGlow" cx="40%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#DFFFFF"/>
      <stop offset="45%" stop-color="#7FF3F7"/>
      <stop offset="100%" stop-color="#0F8A90"/>
    </radialGradient>
    <linearGradient id="headGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="60%" stop-color="#EEF3F8"/>
      <stop offset="100%" stop-color="#CFDBE6"/>
    </linearGradient>
    <linearGradient id="bodyGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2B4A6B"/>
      <stop offset="100%" stop-color="#16283D"/>
    </linearGradient>
    <linearGradient id="visorGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E8536A"/>
      <stop offset="100%" stop-color="#8C0B1F"/>
    </linearGradient>
    <filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="dropShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000000" flood-opacity="0.18"/>
    </filter>
  </defs>

  <g filter="url(#dropShadow)">
    <rect x="8" y="150" width="92" height="60" rx="7" fill="#2B3A4A"/>
    <rect x="15" y="157" width="78" height="42" rx="3" fill="#0B141F"/>
    <rect x="20" y="162" width="50" height="6" rx="3" fill="#7FF3F7" opacity="0.7"/>
    <rect x="20" y="172" width="66" height="6" rx="3" fill="#3A536B"/>
    <rect x="20" y="182" width="40" height="6" rx="3" fill="#3A536B"/>
    <rect x="-2" y="208" width="122" height="8" rx="4" fill="#16283D"/>

    <rect x="115" y="16" width="98" height="34" rx="14" fill="#FFFFFF" stroke="#C8102E" stroke-width="2"/>
    <text x="164" y="38" font-family="Arial" font-size="13" fill="#333333" text-anchor="middle">Bonjour !</text>

    <rect x="128" y="128" width="72" height="62" rx="16" fill="url(#bodyGrad)"/>
    <circle cx="164" cy="159" r="11" fill="url(#eyeGlow)" filter="url(#softGlow)" class="robot-eye"/>
    <rect x="150" y="172" width="28" height="4" rx="2" fill="#0F8A90" opacity="0.6"/>

    <rect x="103" y="143" width="28" height="14" rx="7" fill="#16283D"/>
    <rect x="197" y="143" width="28" height="14" rx="7" fill="#16283D"/>

    <path d="M118 93 Q164 52 210 93" stroke="#16283D" stroke-width="8" fill="none" stroke-linecap="round"/>
    <rect x="110" y="88" width="17" height="27" rx="8.5" fill="#16283D"/>
    <rect x="203" y="88" width="17" height="27" rx="8.5" fill="#16283D"/>
    <circle cx="118.5" cy="101" r="4.5" fill="#7FF3F7" filter="url(#softGlow)"/>
    <circle cx="211.5" cy="101" r="4.5" fill="#7FF3F7" filter="url(#softGlow)"/>

    <circle cx="164" cy="98" r="49" fill="url(#headGrad)" stroke="#16283D" stroke-width="4"/>

    <path d="M120 78 Q164 55 208 78 L208 68 Q164 46 120 68 Z" fill="url(#visorGrad)" opacity="0.9"/>

    <ellipse cx="147" cy="100" rx="13" ry="16" fill="url(#eyeGlow)" filter="url(#softGlow)" class="robot-eye"/>
    <ellipse cx="181" cy="100" rx="13" ry="16" fill="url(#eyeGlow)" filter="url(#softGlow)" class="robot-eye"/>
    <ellipse cx="147" cy="95" rx="4.5" ry="5.5" fill="#FFFFFF"/>
    <ellipse cx="181" cy="95" rx="4.5" ry="5.5" fill="#FFFFFF"/>

    <path d="M149 122 Q164 133 179 122" stroke="#16283D" stroke-width="3.2" fill="none" stroke-linecap="round"/>

    <path d="M121 108 Q108 118 121 129" stroke="#16283D" stroke-width="3" fill="none"/>
    <circle cx="121" cy="129" r="4.5" fill="#C8102E"/>

    <rect x="160" y="42" width="7" height="14" rx="3.5" fill="#16283D"/>
    <circle cx="163.5" cy="40" r="6" fill="#C8102E" class="robot-antenna"/>
  </g>
</svg>
"""

components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
body {{ margin: 0; font-family: 'Inter', Arial, sans-serif; background: transparent; }}
.ags-banner {{
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%); backdrop-filter: blur(10px);
    border-radius: 24px; padding: 26px 36px;
    display: flex; align-items: center; gap: 30px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.45);
    width: 100%; box-sizing: border-box; border: 1px solid rgba(255,255,255,0.12);
}}
.ags-banner::before {{
    content:""; position:absolute; right:-80px; top:-120px; width:380px; height:380px; border-radius:50%;
    background: radial-gradient(circle, rgba(244,63,94,0.55), transparent 65%);
}}
.ags-banner::after {{
    content:""; position:absolute; left:-60px; bottom:-140px; width:300px; height:300px; border-radius:50%;
    background: radial-gradient(circle, rgba(80,140,255,0.22), transparent 65%);
}}
.ags-banner > * {{ position: relative; z-index: 1; }}
.ags-badge {{
    display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:#FF8A9B; background: rgba(200,16,46,0.18); border:1px solid rgba(255,107,129,0.35);
    padding:4px 12px; border-radius:999px; margin-bottom:10px;
}}
.ags-banner-title {{
    color: white; margin: 0; font-size: 2.2rem; font-weight: 800; line-height: 1.15; letter-spacing:-0.02em;
}}
.ags-banner-title span {{ background: linear-gradient(90deg,#FB7185,#F97316); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.ags-banner-text p {{ color: #C9D3E6; margin: 8px 0 14px 0; font-size: 1rem; max-width: 560px; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; }}
.pill {{
    font-size:.78rem; font-weight:600; color:#E8ECF3; padding:5px 12px; border-radius:999px;
    background: rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.16);
}}
.robot-float {{ animation: floatY 3.5s ease-in-out infinite; filter: drop-shadow(0 14px 22px rgba(0,0,0,0.35)); }}
@keyframes floatY {{
    0%   {{ transform: translateY(0px); }}
    50%  {{ transform: translateY(-8px); }}
    100% {{ transform: translateY(0px); }}
}}
.robot-eye {{ animation: pulseGlow 2.2s ease-in-out infinite; }}
@keyframes pulseGlow {{ 0% {{ opacity: 0.75; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.75; }} }}
.robot-antenna {{ animation: blink 1.4s step-start infinite; }}
@keyframes blink {{ 50% {{ opacity: 0.3; }} }}
</style>
<div class="ags-banner">
    <div>{ROBOT_SVG}</div>
    <div class="ags-banner-text">
        <div class="ags-badge">IA générative · RAG</div>
        <div class="ags-banner-title">Assistant IA <span>AGS GROUPE</span></div>
        <p>Votre assistant intelligent pour toutes vos questions sur nos projets, devis et services.</p>
        <div class="pills">
            <span class="pill">📚 Documents internes</span>
            <span class="pill">🌐 FR · AR · Darija</span>
            <span class="pill">🔊 Réponse vocale</span>
            <span class="pill">📷 Analyse de chantier</span>
        </div>
    </div>
</div>
""", height=215)

LANGUAGES = {
    "Français": {
        "code": "fr",
        "instruction": "Réponds en français, de façon claire et professionnelle.",
    },
    "العربية (Arabe standard)": {
        "code": "ar",
        "instruction": "Réponds en arabe standard moderne (فصحى), de façon claire et professionnelle.",
    },
    "الدارجة (Darija marocaine)": {
        "code": "ar",
        "instruction": (
            "Réponds en darija marocaine (arabe dialectal marocain), écrite avec l'alphabet "
            "arabe, comme le parlerait un conseiller commercial marocain à un client. "
            "Reste professionnel et clair."
        ),
    },
}


@st.cache_resource
def load_retriever_and_llm():
    if not os.path.exists(PERSIST_DIR):
        st.error("Base vectorielle introuvable. Lance d'abord: python ingest.py")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Clé GROQ_API_KEY manquante. Ajoute-la dans le fichier .env (voir README).")
        st.stop()

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, groq_api_key=api_key)
    return retriever, llm


@st.cache_resource
def load_image_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def build_chain(retriever, llm, language_instruction):
    prompt_template = f"""Tu es l'assistant IA interne d'AGS GROUPE, entreprise de construction
(villas, riads, bâtiments) basée à Marrakech.

Réponds à la question en priorité à partir du contexte fourni ci-dessous, qui provient
de la documentation officielle de l'entreprise.

Si le contexte ne contient pas assez d'informations pour répondre complètement, tu peux
compléter avec tes connaissances générales sur le secteur de la construction et du BTP,
à condition de préciser clairement que cette partie de la réponse n'est PAS issue de la
documentation officielle d'AGS GROUPE.

Ne mentionne jamais explicitement le mot "contexte" dans ta réponse ; parle plutôt de
"nos documents" ou "notre documentation interne".

{language_instruction}

Contexte:
{{context}}

Question: {{question}}

Réponse:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def analyze_image(image_bytes, llm, user_question):
    """Génère une description brute de l'image (BLIP), puis la reformule en français via le LLM."""
    captioner = load_image_captioner()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_caption = captioner(image)[0]["generated_text"]

    question = user_question.strip() if user_question.strip() else (
        "Décris cette photo de chantier et identifie l'étape de construction probable."
    )

    reformulation_prompt = f"""Tu es un assistant technique pour AGS GROUPE, entreprise de
construction (villas, riads) à Marrakech.

Une analyse automatique d'image (modèle de vision par ordinateur) a produit la description
brute suivante, en anglais, à partir d'une photo de chantier :
"{raw_caption}"

Question posée par l'utilisateur : {question}

En te basant sur cette description automatique, rédige une réponse professionnelle et
claire en français. Précise brièvement que ta réponse s'appuie sur une description
générée automatiquement à partir de l'image, qui peut être imprécise ou incomplète."""

    response = llm.invoke(reformulation_prompt)
    return raw_caption, response.content


FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
USE_UNICODE_FONT = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)


def clean_line_for_pdf(line):
    """Nettoie une ligne markdown : liens, gras, titres, tirets de séparation."""
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"^-{2,}\s*-{0,}$", "", line)
    return line.strip()


class AGSPdf(FPDF):
    def __init__(self):
        super().__init__()
        if USE_UNICODE_FONT:
            self.add_font("Body", "", FONT_REGULAR)
            self.add_font("Body", "B", FONT_BOLD)
            self.font_name = "Body"
        else:
            self.font_name = "Helvetica"

    def safe(self, text):
        if USE_UNICODE_FONT:
            return text
        return text.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        self.set_xy(10, 12)
        self.set_font(self.font_name, style="B", size=15)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, self.safe("AGS GROUPE"))
        self.set_xy(10, 20)
        self.set_font(self.font_name, size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, self.safe("Assistant IA - Compte rendu de conversation"))

        if os.path.exists(LOGO_PATH):
            logo_w = 25
            self.image(LOGO_PATH, x=self.w - self.r_margin - logo_w, y=8, w=logo_w)

        self.set_draw_color(200, 30, 30)
        self.set_line_width(0.6)
        self.line(10, 30, self.w - self.r_margin, 30)
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, self.safe(f"Page {self.page_no()}"), align="C")

    def write_body_text(self, raw_text):
        lines = raw_text.split("\n")
        for line in lines:
            clean = clean_line_for_pdf(line)
            if not clean:
                self.ln(2)
                continue

            self.set_x(self.l_margin)

            if clean.startswith("#"):
                heading = clean.lstrip("#").strip()
                self.set_font(self.font_name, style="B", size=12)
                self.set_text_color(180, 20, 20)
                self.ln(2)
                self.multi_cell(self.epw, 7, self.safe(heading))
                self.set_text_color(20, 20, 20)
                self.ln(1)

            elif clean.startswith("- ") or clean.startswith("* "):
                item = clean[2:].strip()
                self.set_font(self.font_name, size=10)
                self.set_x(self.l_margin + 5)
                self.multi_cell(self.epw - 5, 6, self.safe(f"- {item}"))

            else:
                self.set_font(self.font_name, size=10)
                self.multi_cell(self.epw, 6, self.safe(clean))


def generate_pdf(messages):
    pdf = AGSPdf()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font(pdf.font_name, size=9)
    pdf.set_text_color(120, 120, 120)
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    pdf.cell(0, 6, pdf.safe(f"Genere le {date_str}"), ln=True)
    pdf.ln(6)

    for msg in messages:
        is_user = msg["role"] == "user"
        label = "Question du client" if is_user else "Reponse de l'assistant"

        pdf.set_fill_color(230, 230, 230) if is_user else pdf.set_fill_color(255, 235, 235)

        pdf.set_x(pdf.l_margin)
        pdf.set_font(pdf.font_name, style="B", size=10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 8, pdf.safe(label), ln=True, fill=True)
        pdf.ln(1)

        pdf.write_body_text(msg["content"])
        pdf.ln(4)

    return bytes(pdf.output())


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.divider()
    st.header("À propos")
    st.write(
        "Ce chatbot utilise une architecture RAG (Retrieval-Augmented Generation) : "
        "il recherche les passages pertinents dans les documents internes de l'entreprise "
        "puis génère une réponse à partir de ces passages, avec citation des sources."
    )
    st.write("**Exemples de questions :**")
    st.markdown(
        "- Quel est le délai moyen pour une villa de 300m² ?\n"
        "- Quel acompte dois-je verser à la signature ?\n"
        "- Quelles garanties propose AGS GROUPE ?\n"
        "- Quelles sont les normes parasismiques appliquées ?"
    )
    st.divider()
    selected_language = st.selectbox("🌐 Langue de réponse", list(LANGUAGES.keys()))
    st.divider()

    if st.session_state.messages:
        pdf_bytes = generate_pdf(st.session_state.messages)
        st.download_button(
            label="📄 Télécharger la conversation (PDF)",
            data=pdf_bytes,
            file_name="conversation_ags_groupe.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("Le bouton PDF apparaîtra ici après votre première question.")

    st.divider()
    if st.button("🚪 Déconnexion"):
        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

retriever, llm = load_retriever_and_llm()

tab_chat, tab_image = st.tabs(["💬 Assistant textuel", "📷 Analyse de photo de chantier"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

    if not st.session_state.messages:
        st.markdown('<div class="sugg-title">👋 Bonjour, comment puis-je vous aider ?</div>'
                    '<div class="sugg-sub">Choisissez une question ou écrivez la vôtre ci-dessous.</div>',
                    unsafe_allow_html=True)
        _suggestions = [
            "⏱️ Quel est le délai moyen pour une villa de 300 m² ?",
            "💰 Quel acompte dois-je verser à la signature ?",
            "🛡️ Quelles garanties propose AGS GROUPE ?",
            "🏗️ Quelles normes parasismiques sont appliquées ?",
        ]
        _cols = st.columns(2)
        for _i, _q in enumerate(_suggestions):
            with _cols[_i % 2]:
                st.markdown('<div class="sugg">', unsafe_allow_html=True)
                if st.button(_q, key=f"sugg_{_i}", use_container_width=True):
                    st.session_state.pending_question = _q.split(" ", 1)[1]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    question = st.chat_input("Posez votre question...")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                lang_config = LANGUAGES[selected_language]
                chain = build_chain(retriever, llm, lang_config["instruction"])
                result = chain.invoke({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

            st.markdown(answer)

            with st.spinner("Génération de la voix..."):
                tts = gTTS(text=answer, lang=lang_config["code"])
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_data = audio_buffer.getvalue()
                st.audio(audio_data, format="audio/mp3")

            if sources:
                with st.expander("📄 Sources utilisées"):
                    seen = set()
                    for doc in sources:
                        src = os.path.basename(doc.metadata.get("source", "inconnu"))
                        if src not in seen:
                            seen.add(src)
                            st.markdown(f"**{src}**")
                            st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_data})
        st.rerun()

with tab_image:
    st.markdown(
        '<div class="info-card"><b>📷 Analyse de photo de chantier</b><br>'
        "Envoyez une photo de chantier (par exemple issue d'un rapport journalier) "
        "et l'assistant en décrit l'avancement et les points d'attention.</div>",
        unsafe_allow_html=True,
    )
    uploaded_image = st.file_uploader(
        "Choisis une photo (JPG, PNG)", type=["jpg", "jpeg", "png"]
    )
    image_question = st.text_input(
        "Question sur la photo (facultatif)",
        placeholder="Ex : à quelle étape de construction correspond cette photo ?",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Photo envoyée", use_container_width=True)

        if st.button("🔍 Analyser cette photo"):
            with st.spinner("Analyse de l'image en cours (peut prendre 1-2 min la 1ère fois)..."):
                image_bytes = uploaded_image.getvalue()
                raw_caption, analysis = analyze_image(image_bytes, llm, image_question)
            st.markdown("### Résultat de l'analyse")
            st.markdown(analysis)
            with st.expander("🔧 Détail technique (description brute générée par le modèle vision)"):
                st.code(raw_caption)