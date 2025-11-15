class BaseInterpreter:
    def interpret(self, chart_context: dict) -> dict:
        raise NotImplementedError("Subclasses must implement interpret()")

