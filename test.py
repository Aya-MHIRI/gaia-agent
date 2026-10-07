import json
import os
import time

import requests

from agent_smol import answer
#from agent_langgraph import answer_vote as answer
from dotenv import load_dotenv
load_dotenv()
USERNAME = os.environ["HF_USERNAME"]
SPACE_CODE = os.environ["SPACE_CODE"]
SUBMIT = False  # passe à True quand tu as relu answers.json

questions = requests.get(f"{API}/questions", timeout=120).json()
os.makedirs("files", exist_ok=True)

# Reprise : on recharge les réponses déjà obtenues (non vides)
done = {}
if os.path.exists("answers.json"):
    for item in json.load(open("answers.json", encoding="utf-8")):
        if item["submitted_answer"]:
            done[item["task_id"]] = item["submitted_answer"]

results = []
for q in questions:
    tid = q["task_id"]
    if tid in done:
        print(tid, "(déjà fait) ->", done[tid])
        results.append({"task_id": tid, "submitted_answer": done[tid]})
        continue

    path = None
    # Vérification souple de la présence d'un fichier joint dans GAIA
    file_name = q.get("file_name") or q.get("filename")
    
    if file_name:
        try:
            r = requests.get(f"{API}/files/{tid}", timeout=60)
            r.raise_for_status()
            path = os.path.join("files", file_name)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"Fichier téléchargé : {path}")
        except Exception as e:
            print("Téléchargement du fichier impossible :", e)
            path = None

    try:
        ans = answer(q["question"], path)
    except Exception as e:
        ans = ""
        print("Erreur :", e)

    print(tid, "->", ans)
    results.append({"task_id": tid, "submitted_answer": ans})
    json.dump(results + [], open("answers.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    time.sleep(2)

# Relance le script pour retenter les réponses vides avant de soumettre.
if SUBMIT:
    payload = {"username": USERNAME, "agent_code": SPACE_CODE, "answers": results}
    print(requests.post(f"{API}/submit", json=payload, timeout=60).json())
