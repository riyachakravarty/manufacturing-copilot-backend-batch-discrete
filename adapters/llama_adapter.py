import requests
from .base import BaseInterpreter
from utils.response_parser import parse_llm_json
from app.config import settings

class LlamaInterpreter(BaseInterpreter):

    def interpret(self, chart_context: dict) -> dict:
        prompt = chart_context["prompt"]

        payload = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.2
        }

        resp = requests.post(
            f"{settings.MODEL_ENDPOINT}/generate",
            json=payload,
            timeout=settings.TIMEOUT,
        )

        resp.raise_for_status()
        result = resp.json()

        return parse_llm_json(result["text"])

