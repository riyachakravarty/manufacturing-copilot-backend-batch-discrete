import json
import re

def parse_llm_json(text: str):
    """
    Safely parse JSON returned by LLM.
    If the LLM returns text instead of JSON, wrap it into a dict.
    """
    if not text:
        return {"raw_response": ""}

    # Extract JSON block between braces
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass  # fallback below

    return {"raw_response": text}


