from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    qdrant_url: str
    app_env: str = "development"
    
    class Config:
        env_file = ".env"

settings = Settings()
