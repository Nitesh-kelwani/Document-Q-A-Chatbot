from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8-sig',
        case_sensitive=False,
        extra='ignore',
    )

    app_name: str = 'Document Q&A Chatbot'
    api_prefix: str = '/api'

    documents_dir: Path = BASE_DIR / 'data' / 'documents'
    vectorstore_dir: Path = BASE_DIR / 'data' / 'vectorstore'

    chunk_size: int = Field(default=1000, alias='CHUNK_SIZE')
    chunk_overlap: int = Field(default=150, alias='CHUNK_OVERLAP')
    retrieval_k: int = Field(default=4, alias='RETRIEVAL_K')
    temperature: float = Field(default=0.1, alias='TEMPERATURE')
    max_response_tokens: int | None = Field(default=700, alias='MAX_RESPONSE_TOKENS')

    azure_openai_endpoint: str = Field(alias='AZURE_OPENAI_ENDPOINT')
    azure_openai_api_key: str = Field(alias='AZURE_OPENAI_API_KEY')
    azure_openai_api_version: str = Field(default='2024-10-21', alias='OPENAI_API_VERSION')
    azure_openai_chat_deployment: str = Field(alias='AZURE_OPENAI_CHAT_DEPLOYMENT')
    azure_openai_chat_model: str | None = Field(default=None, alias='AZURE_OPENAI_CHAT_MODEL')
    azure_openai_embedding_deployment: str = Field(alias='AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
    azure_openai_embedding_model: str | None = Field(default='text-embedding-ada-002', alias='AZURE_OPENAI_EMBEDDING_MODEL')

    api_base_url: str = Field(default='http://localhost:8000/api', alias='API_BASE_URL')

    def ensure_directories(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
