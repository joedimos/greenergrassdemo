from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional

class Settings(BaseSettings):
    # Required settings
    database_url: str
    openai_api_key: str
    qdrant_url: str
    
    # Optional settings with defaults
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    search_k: int = 4
    
    # If you need an email field, validate it manually
    admin_email: Optional[str] = None
    
    # Pydantic configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator('admin_email')
    @classmethod
    def validate_admin_email(cls, v):
        """Simple email validation without external dependencies"""
        if v is None:
            return v
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

settings = Settings()

# Validate critical settings
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY is required")

if not settings.qdrant_url:
    raise ValueError("QDRANT_URL is required")

if not settings.database_url:
    raise ValueError("DATABASE_URL is required")
