from adapters.factory import load_interpreter
from prompt_builder import (
    build_shap_summary_prompt,
    build_shap_dependence_prompt
)


def interpret_shap_summary(context: dict):
    interpreter = load_interpreter()
    prompt = build_shap_summary_prompt(context)
    return interpreter.interpret({"prompt": prompt})


def interpret_shap_dependence(context: dict):
    interpreter = load_interpreter()
    prompt = build_shap_dependence_prompt(context)
    return interpreter.interpret({"prompt": prompt})
