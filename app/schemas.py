from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    selected_documents: list[str] = Field(default_factory=list)


class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    retrieved_chunks: int


class IngestionResponse(BaseModel):
    message: str
    files_indexed: int
    chunks_indexed: int


class DocumentListResponse(BaseModel):
    documents: list[str]
