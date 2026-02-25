import json
import urllib.request
from typing import Optional

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def ollama_generate(prompt: str, model: str = "llama3.1:8b", temperature: float = 0.2, timeout_s: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return out.get("response", "").strip()
