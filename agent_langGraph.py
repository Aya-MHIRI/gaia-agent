import os
import re
import time
from collections import Counter
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from smolagents import DuckDuckGoSearchTool

# On réutilise tes outils déjà écrits
from agent_smol import (fetch_page, wikipedia_search, read_file,
                        run_python, youtube_transcript)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ================= OUTILS =================
_search = DuckDuckGoSearchTool()


@lc_tool
def web_search(query: str) -> str:
    """Recherche sur le web (DuckDuckGo)."""
    return str(_search(query=query))


@lc_tool
def wiki_search(query: str) -> str:
    """Recherche des articles Wikipedia en anglais."""
    return str(wikipedia_search(query=query))


@lc_tool
def page(url: str, start: int = 0, keyword: str = "") -> str:
    """Lit une page web. 'keyword' saute au bon passage, 'start' lit la suite."""
    return str(fetch_page(url=url, start=start, keyword=keyword))


@lc_tool
def file_reader(path: str) -> str:
    """Lit un fichier joint (Excel, CSV, audio transcrit, texte)."""
    return str(read_file(path=path))


@lc_tool
def python_runner(path: str) -> str:
    """Exécute un fichier Python joint et renvoie sa sortie."""
    return str(run_python(path=path))


@lc_tool
def youtube(url: str) -> str:
    """Récupère la transcription d'une vidéo YouTube."""
    return str(youtube_transcript(url=url))


@lc_tool
def calculator(expression: str) -> str:
    """Calcule une expression arithmétique (chiffres, + - * / ( ) et points)."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "Expression invalide."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Erreur de calcul : {e}"


ALL_TOOLS = [web_search, wiki_search, page, file_reader,
             python_runner, youtube, calculator]

# Chaque route n'expose que les outils utiles (moins de confusion pour le LLM)
TOOLSETS = {
    "audio":   [file_reader, calculator],
    "excel":   [file_reader, calculator],
    "python":  [file_reader, python_runner, calculator],
    "image":   [file_reader],
    "youtube": [youtube, web_search, wiki_search, page],
    "web":     [web_search, wiki_search, page, calculator],
}

HINTS = {
    "audio":   "Lis le fichier audio avec file_reader (il renvoie la transcription).",
    "excel":   "Lis le fichier avec file_reader, puis calcule avec calculator si besoin.",
    "python":  "Lis le fichier avec file_reader, puis exécute-le avec python_runner.",
    "image":   "Lis le fichier avec file_reader (description de l'image).",
    "youtube": "Utilise l'outil youtube pour obtenir la transcription de la vidéo.",
    "web":     "Cherche avec web_search ou wiki_search, puis lis les pages avec page (keyword).",
}

# ================= MODELE =================
llm = ChatOpenAI(
    model=os.environ["LG_MODEL"],
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    temperature=0.2,
)
_bound = {}

SYSTEM = """Tu résous des questions de type GAIA.
- N'invente jamais de données : appuie-toi sur tes outils.
- Quand tu as la réponse, écris UNIQUEMENT la réponse : sans phrase, sans explication,
  sans 'FINAL ANSWER', au format exact demandé (un nombre reste un nombre,
  une liste a ses éléments séparés par une virgule et un espace)."""


# ================= UTILITAIRES =================
def clean(s: str) -> str:
    s = re.sub(r"(?is)<think>.*?</think>", "", str(s)).strip()
    s = re.sub(r"(?i)^\s*final answer\s*[:\-]?\s*", "", s)
    return re.sub(r"\.$", "", s.strip().strip('"').strip("'").strip())


def detect_route(question: str, file_path: str) -> str:
    ext = os.path.splitext(file_path or "")[1].lower()
    if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
        return "audio"
    if ext in (".xlsx", ".xls", ".csv"):
        return "excel"
    if ext == ".py":
        return "python"
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        return "image"
    if re.search(r"youtube\.com|youtu\.be", question):
        return "youtube"
    return "web"


COMMON = {"the", "of", "and", "to", "in", "is", "you", "if", "a", "as", "what", "word"}


def looks_reversed(text: str) -> bool:
    fwd = sum(w in COMMON for w in re.findall(r"[a-z]+", text.lower()))
    rev = sum(w in COMMON for w in re.findall(r"[a-z]+", text[::-1].lower()))
    return rev >= 2 and rev > 2 * fwd


# ================= ETAT =================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    file_path: str
    route: str
    attempts: int
    retry: bool


# ================= NOEUDS =================
def route(state: State):
    q = state["question"]
    r = detect_route(q, state["file_path"])
    note = ""
    if looks_reversed(q):
        q = q[::-1]
        note = "NB : la question était écrite à l'envers ; la voici remise à l'endroit.\n"
    prompt = f"{note}{q}\n\n{HINTS[r]}"
    if state["file_path"]:
        prompt += f"\n[FICHIER JOINT] : '{state['file_path']}'"
    return {"route": r, "question": q, "messages": [HumanMessage(content=prompt)]}


def assistant(state: State):
    r = state["route"]
    if r not in _bound:
        _bound[r] = llm.bind_tools(TOOLSETS[r])
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    for i in range(3):
        try:
            return {"messages": [_bound[r].invoke(msgs)]}
        except Exception as e:
            print("Erreur LLM :", e)
            time.sleep(10 * (i + 1))
    raise RuntimeError("LLM indisponible")


def validate(state: State):
    text = clean(state["messages"][-1].content)
    bad = (not text) or len(text.split()) > 15 or "\n" in text or text.endswith("?")
    if bad and state["attempts"] < 2:
        return {
            "attempts": state["attempts"] + 1,
            "retry": True,
            "messages": [HumanMessage(content=(
                "Ta réponse n'est pas au bon format. Si tu as besoin d'une information, "
                "utilise un outil. Sinon réponds UNIQUEMENT par la réponse finale, "
                "sans phrase ni explication."))],
        }
    return {"retry": False}


def check(state: State):
    cand = clean(state["messages"][-1].content)
    r = llm.invoke([
        SystemMessage(content=(
            "Vérifie que la réponse candidate respecte toutes les contraintes de forme "
            "de la question (ordre, décimales, unités, liste, singulier/pluriel, "
            "abréviations, nombre d'éléments). Si oui, répète-la à l'identique. "
            "Sinon corrige uniquement la forme, sans changer le fond. "
            "Réponds UNIQUEMENT par la réponse.")),
        HumanMessage(content=f"Question : {state['question']}\n\nRéponse candidate : {cand}"),
    ])
    return {"messages": [AIMessage(content=clean(r.content) or cand)]}


# ================= GRAPHE =================
g = StateGraph(State)
g.add_node("route", route)
g.add_node("assistant", assistant)
g.add_node("tools", ToolNode(ALL_TOOLS))
g.add_node("validate", validate)
g.add_node("check", check)

g.add_edge(START, "route")
g.add_edge("route", "assistant")
g.add_conditional_edges("assistant", tools_condition,
                        {"tools": "tools", END: "validate"})
g.add_edge("tools", "assistant")
g.add_conditional_edges("validate", lambda s: "assistant" if s["retry"] else "check",
                        {"assistant": "assistant", "check": "check"})
g.add_edge("check", END)
app = g.compile()


# ================= FONCTIONS PRINCIPALES =================
def answer(question: str, file_path: str = None) -> str:
    out = app.invoke(
        {"messages": [], "question": question, "file_path": file_path or "",
         "route": "web", "attempts": 0, "retry": False},
        config={"recursion_limit": 30},
    )
    return clean(out["messages"][-1].content)


def answer_vote(question: str, file_path: str = None, n: int = 3) -> str:
    """Lance n essais et garde la réponse la plus fréquente."""
    answers = []
    for _ in range(n):
        try:
            a = answer(question, file_path).strip()
            if a:
                answers.append(a)
        except Exception as e:
            print("Essai échoué :", e)
    if not answers:
        return ""
    best, _ = Counter(a.lower() for a in answers).most_common(1)[0]
    return next(a for a in answers if a.lower() == best)