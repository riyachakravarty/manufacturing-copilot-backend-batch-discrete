from pydantic-settings import BaseSettings

class Settings(BaseSettings):
    MODEL_TYPE: str = "llama"       # llama / mistral / phi / rule
    MODEL_ENDPOINT: str = "http://127.0.0.1:8002"
    TIMEOUT: int = 30

settings = Settings()
