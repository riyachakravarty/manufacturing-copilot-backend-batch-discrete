from app.config import settings
from .llama_adapter import LlamaInterpreter
from .mistral_adapter import MistralInterpreter
from .phi_adapter import Phi3Interpreter
from .rule_based_adapter import RuleBasedInterpreter

def load_interpreter():

    model_type = settings.MODEL_TYPE.lower()

    if model_type == "llama":
        return LlamaInterpreter()

    if model_type == "mistral":
        from my_mistral_loader import mistral_model
        return MistralInterpreter(mistral_model)

    if model_type == "phi":
        from my_phi_loader import phi_model
        return Phi3Interpreter(phi_model)

    if model_type == "rule":
        return RuleBasedInterpreter()

    raise ValueError(f"Unsupported model type: {model_type}")

