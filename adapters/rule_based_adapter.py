from .base import BaseInterpreter

class RuleBasedInterpreter(BaseInterpreter):

    def interpret(self, chart_context: dict) -> dict:
        prompt = chart_context.get("prompt", "")

        if "SHAP SUMMARY" in prompt:
            return {
                "insight": "Top features have the highest mean |SHAP| values.",
                "confidence": 0.80,
                "suggested_next_steps": [
                    "Check stability of top features",
                    "Investigate SHAP dependence for top 3"
                ]
            }

        if "SHAP DEPENDENCE" in prompt:
            return {
                "insight": "Feature shows a monotonic relationship.",
                "confidence": 0.75,
                "suggested_next_steps": [
                    "Investigate interaction with top drivers"
                ]
            }

        return {
            "insight": "Unable to analyze context.",
            "confidence": 0.50,
            "suggested_next_steps": []
        }

