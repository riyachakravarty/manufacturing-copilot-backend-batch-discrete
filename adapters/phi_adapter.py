from .base import BaseInterpreter
from utils.response_parser import parse_llm_json

class Phi3Interpreter(BaseInterpreter):

    def __init__(self, model):
        self.model = model

    def interpret(self, chart_context: dict) -> dict:
        prompt = chart_context["prompt"]
        output = self.model(prompt)
        return parse_llm_json(output)

