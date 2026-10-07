import json
import requests

API = "https://agents-course-unit4-scoring.hf.space"
import os
from dotenv import load_dotenv
load_dotenv()
USERNAME = os.environ["HF_USERNAME"]
SPACE_CODE = os.environ["SPACE_CODE"]

answers = json.load(open("answers.json", encoding="utf-8"))
payload = {"username": USERNAME, "agent_code": SPACE_CODE, "answers": answers}

r = requests.post(f"{API}/submit", json=payload, timeout=120)
print(r.status_code)
print(r.json())