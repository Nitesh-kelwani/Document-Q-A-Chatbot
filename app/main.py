from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version='0.1.0',
    description='Document Q&A chatbot powered by LangChain and Azure OpenAI.',
)
app.include_router(router, prefix=settings.api_prefix)


@app.get('/', tags=['meta'])
def root() -> dict[str, str]:
    return {
        'message': 'Document Q&A chatbot is running.',
        'docs_url': '/docs',
    }
