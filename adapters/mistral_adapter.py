from .base import BaseInterpreter
from app.utils.response_parser import parse_llm_json

class MistralInterpreter(BaseInterpreter):

    def __init__(self, model):
        self.model = model   # local mistral model

    def interpret(self, chart_context: dict) -> dict:
        prompt = chart_context["prompt"]
        output = self.model(prompt)   # e.g., vLLM or transformers
        return parse_llm_json(output)

