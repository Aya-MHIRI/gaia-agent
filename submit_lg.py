import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://agents-course-unit4-scoring.hf.space"
USERNAME = os.environ["HF_USERNAME"]
SPACE_CODE = os.environ["SPACE_CODE"]

answers = json.load(open("answers_lg.json", encoding="utf-8"))
empty = sum(1 for a in answers if not a["submitted_answer"].strip())
print(f"{len(answers)} réponses, dont {empty} vides")

if input("Soumettre ? (o/n) ").strip().lower() != "o":
    raise SystemExit("Annulé.")

payload = {"username": USERNAME, "agent_code": SPACE_CODE, "answers": answers}
r = requests.post(f"{API}/submit", json=payload, timeout=120)
print(r.status_code)
print(r.json())