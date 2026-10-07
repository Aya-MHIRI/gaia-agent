import base64
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# 1. CONFIGURATION DES DOSSIERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
TEMP_DIR = BASE_DIR / "temp"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION DES CACHES
# ============================================================

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)


# ============================================================
# 3. LIMITATION DES THREADS
# ============================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================
# 4. IMPORTS
# ============================================================

import pandas as pd
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify

from huggingface_hub import InferenceClient

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    tool,
)



# ============================================================
# 5. CHARGEMENT DU .ENV
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN est absent du fichier .env"
    )


# ============================================================
# 6. MODÈLE HUGGING FACE
# ============================================================
MODEL_ID = os.getenv("HF_MODEL_ID", "")
if not MODEL_ID:
    raise RuntimeError("HF_MODEL_ID est absent du fichier .env")

model = InferenceClientModel(
    model_id=MODEL_ID,
    token=HF_TOKEN,
)


# ============================================================
# 7. CONFIGURATION GÉNÉRALE
# ============================================================

HEADERS = {
    "User-Agent": "gaia-agent-student/1.0 (student project)"
}

CHUNK = 3000


# ============================================================
# 8. OUTIL : TÉLÉCHARGER UNE PAGE WEB
# ============================================================

@tool
def fetch_page(url: str, start: int = 0, keyword: str = "") -> str:
    """
    Télécharge une page web et renvoie son contenu utile au format Markdown.

    Args:
        url: adresse complète de la page.
        start: position de départ dans le texte.
        keyword: mot-clé à rechercher.
    """

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        r.raise_for_status()

    except Exception as e:
        return f"Erreur de téléchargement : {e}"

    soup = BeautifulSoup(
        r.text,
        "html.parser"
    )

    # Suppression des éléments inutiles
    for t in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "noscript",
            "form"
        ]
    ):
        t.decompose()

    main = (
        soup.select_one("#mw-content-text")
        or soup.find("main")
        or soup.body
        or soup
    )

    for t in main.select(
        ".navbox, .mw-editsection, .reflist, "
        ".references, .sidebar, .mw-references-wrap"
    ):
        t.decompose()

    text = markdownify(
        str(main),
        strip=["a", "img"]
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    if keyword:
        idx = text.lower().find(
            keyword.lower()
        )

        if idx < 0:
            return (
                f"Mot-clé '{keyword}' introuvable. "
                f"Taille totale : {len(text)} chars."
            )

        start = max(
            0,
            idx - 200
        )

    chunk = text[
        start:start + CHUNK
    ]

    if start + CHUNK < len(text):
        chunk += (
            f"\n\n[... suite disponible "
            f"avec start={start + CHUNK} ...]"
        )

    return chunk


# ============================================================
# 9. OUTIL : RECHERCHE WIKIPEDIA
# ============================================================

@tool
def wikipedia_search(query: str) -> str:
    """
    Recherche des articles sur Wikipedia en anglais.

    Args:
        query: mots-clés de recherche.
    """

    try:

        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "format": "json",
            },
            headers=HEADERS,
            timeout=20,
        )

        r.raise_for_status()

        hits = r.json()["query"]["search"]

    except Exception as e:
        return f"Erreur Wikipedia : {e}"

    out = []

    for h in hits:

        title = h["title"]

        url = (
            "https://en.wikipedia.org/wiki/"
            + title.replace(" ", "_")
        )

        snippet = re.sub(
            r"<.*?>",
            "",
            h["snippet"]
        )

        out.append(
            f"- {title} : {url}\n"
            f"  {snippet}"
        )

    return "\n".join(out) or "Aucun résultat."


# ============================================================
# 10. OUTIL : TRANSCRIPTION AUDIO
# ============================================================

def _transcribe(path: str) -> str:

    # Groq reste utilisé uniquement pour l'audio.
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        return (
            "Erreur : GROQ_API_KEY est absente du fichier .env "
            "pour la transcription audio."
        )

    try:

        with open(path, "rb") as f:

            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",

                headers={
                    "Authorization": f"Bearer {groq_key}"
                },

                files={
                    "file": (
                        os.path.basename(path),
                        f
                    )
                },

                data={
                    "model": "whisper-large-v3",
                    "response_format": "text"
                },

                timeout=120
            )

        r.raise_for_status()

        return r.text

    except Exception as e:

        return (
            f"Erreur de transcription audio : {e}"
        )


# ============================================================
# 11. OUTIL : DESCRIPTION D'IMAGE
# ============================================================

def _describe_image(path: str) -> str:

    if not HF_TOKEN:
        return (
            "Erreur : HF_TOKEN est absent "
            "du fichier .env"
        )

    try:

        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN
        )

        # Détection du type d'image
        ext = (
            Path(path)
            .suffix
            .lower()
            .lstrip(".")
        )

        if ext in ("jpg", "jpeg"):
            mime = "jpeg"
        elif ext == "png":
            mime = "png"
        elif ext == "webp":
            mime = "webp"
        elif ext == "gif":
            mime = "gif"
        else:
            mime = "jpeg"

        # Lecture + encodage Base64
        with open(path, "rb") as f:
            image_data = f.read()

        image_base64 = base64.b64encode(
            image_data
        ).decode()

        response = client.chat_completion(

            model="Qwen/Qwen2-VL-7B-Instruct",

            messages=[
                {
                    "role": "user",

                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Décris cette image et lis "
                                "tout le texte visible."
                            )
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": (
                                    f"data:image/{mime};base64,"
                                    f"{image_base64}"
                                )
                            }
                        }
                    ]
                }
            ],

            max_tokens=1000
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"Erreur de description de l'image : {e}"
        )


# ============================================================
# 12. OUTIL : LIRE UN FICHIER
# ============================================================

@tool
def read_file(path: str) -> str:
    """
    Lit le contenu d'un fichier local
    (Excel, CSV, Image, Audio, Texte).

    Args:
        path: chemin vers le fichier.
    """

    p = path.lower()

    try:

        # Excel
        if p.endswith(
            (".xlsx", ".xls")
        ):

            sheets = pd.read_excel(
                path,
                sheet_name=None
            )

            text = "\n\n".join(
                f"Feuille {name} :\n"
                f"{df.to_string()}"
                for name, df in sheets.items()
            )

        # CSV
        elif p.endswith(".csv"):

            text = pd.read_csv(
                path
            ).to_string()

        # Audio
        elif p.endswith(
            (
                ".mp3",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac"
            )
        ):

            text = (
                "TRANSCRIPTION AUDIO :\n"
                + _transcribe(path)
            )

        # Images
        elif p.endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif"
            )
        ):

            text = (
                "DESCRIPTION IMAGE :\n"
                + _describe_image(path)
            )

        # Texte
        else:

            text = open(
                path,
                encoding="utf-8",
                errors="ignore"
            ).read()

    except Exception as e:

        return (
            f"Erreur de lecture : {e}"
        )

    return text[:6000]


# ============================================================
# 13. OUTIL : EXÉCUTER PYTHON
# ============================================================

@tool
def run_python(path: str) -> str:
    """
    Exécute un fichier Python et retourne la sortie.

    Args:
        path: chemin vers le fichier .py.
    """

    try:

        r = subprocess.run(
            [
                sys.executable,
                path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        return (
            r.stdout + r.stderr
        )[-4000:] or "(aucune sortie)"

    except Exception as e:

        return f"Erreur : {e}"


# ============================================================
# 14. OUTIL : TRANSCRIPTION YOUTUBE
# ============================================================

@tool
def youtube_transcript(url: str) -> str:
    """
    Récupère la transcription d'une vidéo YouTube.

    Args:
        url: lien de la vidéo YouTube.
    """

    m = re.search(
        r"(?:v=|youtu\.be/)([\w-]{11})",
        url
    )

    if not m:
        return "URL YouTube invalide."

    vid = m.group(1)

    try:

        from youtube_transcript_api import (
            YouTubeTranscriptApi
        )

        data = (
            YouTubeTranscriptApi
            .get_transcript(vid)
        )

        return " ".join(
            s["text"]
            for s in data
        )[:6000]

    except Exception as e:

        return (
            f"Transcription non disponible : {e}"
        )


# ============================================================
# 15. OUTILS DE L'AGENT
# ============================================================

tools = [
    DuckDuckGoSearchTool(),
    wikipedia_search,
    fetch_page,
    read_file,
    run_python,
    youtube_transcript,
]


# ============================================================
# 16. INSTRUCTIONS DE L'AGENT
# ============================================================

INSTRUCTIONS = """
You are an AI assistant executing tasks step-by-step.

Always write Python code inside <code> and </code> tags.

Example step:

Thought: I need to search Wikipedia.

<code>
results = wikipedia_search(query="Mercedes Sosa discography")
print(results)
</code>

When you have the final answer, call final_answer:

<code>
final_answer("4")
</code>
"""


# ============================================================
# 17. CRÉATION DE L'AGENT
# ============================================================

agent = CodeAgent(
    tools=tools,
    model=model,
    instructions=INSTRUCTIONS,
    max_steps=10,
    additional_authorized_imports=[
        "pandas",
        "requests",
        "bs4",
        "re",
        "json"
    ],
)


# ============================================================
# 18. NETTOYAGE DE LA RÉPONSE
# ============================================================

def clean_answer(s: str) -> str:

    s = str(s).strip()

    s = re.sub(
        r"(?i)^\s*final answer\s*[:\-]?\s*",
        "",
        s
    )

    s = (
        s.strip()
        .strip('"')
        .strip("'")
        .strip()
    )

    s = re.sub(
        r"\.$",
        "",
        s
    )

    return s


# ============================================================
# 19. FONCTION PRINCIPALE
# ============================================================

def answer(
    question: str,
    file_path: str = None
) -> str:

    full_prompt = question

    if file_path and os.path.exists(file_path):

        full_prompt += (
            "\n\n[ATTACHED FILE]: "
            "A file relevant to this question is saved "
            f"at path: '{file_path}'. "
            "You MUST inspect or process this file "
            "using your tools."
        )

    res = agent.run(
        full_prompt
    )

    return clean_answer(res)