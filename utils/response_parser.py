import json

def parse_llm_json(text: str):
    """
    Safely parse JSON returned by LLM.
    If the LLM returns text instead of JSON, wrap it into a dict.
    """
    if not text:
        return {"error": "Empty LLM response"}

    try:
        return json.loads(text)
    except Exception:
        # If the LLM response isn't valid JSON, return raw text
        return {"raw_response": text}

