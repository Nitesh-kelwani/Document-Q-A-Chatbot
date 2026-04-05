from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.core.config import Settings, get_settings
from app.schemas import ChatTurn, SourceDocument
from app.services.document_service import DocumentService


@dataclass
class IngestionResult:
    files_indexed: int
    chunks_indexed: int


@dataclass
class AnswerResult:
    answer: str
    sources: list[SourceDocument]
    retrieved_chunks: int


class QAService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.document_service = DocumentService(settings)
        self._lock = Lock()
        self._embeddings = AzureOpenAIEmbeddings(
            model=(
                settings.azure_openai_embedding_model
                or settings.azure_openai_embedding_deployment
            ),
            azure_deployment=settings.azure_openai_embedding_deployment,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )
        self._llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_chat_deployment,
            model=settings.azure_openai_chat_model,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_response_tokens,
        )

    @property
    def documents_dir(self) -> Path:
        return self.settings.documents_dir

    def list_documents(self) -> list[str]:
        return [path.name for path in self.document_service.discover_files()]

    def reindex_corpus(self) -> IngestionResult:
        with self._lock:
            documents = self.document_service.load_documents()
            if not documents:
                raise ValueError(
                    'No PDF files were found in data/documents. Upload a PDF first.'
                )

            chunks = self.document_service.split_documents(documents)
            vector_store = FAISS.from_documents(chunks, self._embeddings)
            vector_store.save_local(str(self.settings.vectorstore_dir))

            return IngestionResult(
                files_indexed=len(self.document_service.discover_files()),
                chunks_indexed=len(chunks),
            )

    def answer_question(
        self,
        question: str,
        history: list[ChatTurn],
        selected_documents: list[str],
    ) -> AnswerResult:
        selected_documents = sorted(set(selected_documents))
        if not selected_documents:
            raise ValueError('Select at least one indexed PDF before asking a question.')

        available_documents = set(self.list_documents())
        unknown_documents = [
            document for document in selected_documents if document not in available_documents
        ]
        if unknown_documents:
            raise ValueError(
                f"Unknown document selection: {', '.join(unknown_documents)}"
            )

        retriever = self._load_retriever(selected_documents)
        agent_executor = self._build_agent_executor(retriever, selected_documents)
        chat_history = self._build_chat_history(history)

        result = agent_executor.invoke(
            {
                'input': question,
                'chat_history': chat_history,
            }
        )
        answer = result['output']

        source_docs = retriever.invoke(question)
        sources = [self._to_source_document(doc) for doc in source_docs]

        return AnswerResult(
            answer=answer,
            sources=sources,
            retrieved_chunks=len(source_docs),
        )

    def _load_retriever(self, selected_documents: list[str]):
        if not self._index_exists():
            raise FileNotFoundError(
                'No vector index found. Upload a PDF or call /api/documents/reindex first.'
            )

        vector_store = FAISS.load_local(
            str(self.settings.vectorstore_dir),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store.as_retriever(
            search_kwargs={
                'k': self.settings.retrieval_k,
                'filter': self._build_document_filter(selected_documents),
            }
        )

    def _build_agent_executor(
        self,
        retriever,
        selected_documents: list[str],
    ) -> AgentExecutor:
        document_prompt = PromptTemplate.from_template(
            'Source: {source}\nPage: {page}\nContent: {page_content}'
        )
        tools = [
            create_retriever_tool(
                retriever,
                name='search_selected_documents',
                description=(
                    'Search only within the currently selected uploaded PDF documents '
                    'for passages relevant to the user question. Use this before '
                    'answering questions about document content.'
                ),
                document_prompt=document_prompt,
            )
        ]

        selected_document_list = ', '.join(selected_documents)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    'You are a helpful document Q&A assistant. '
                    'Only answer from the selected uploaded PDFs. '
                    'Use the search_selected_documents tool before answering document '
                    'questions. If the tool output does not contain the answer, say you '
                    'do not know. When possible, cite the source file and page. '
                    f'Selected documents: {selected_document_list}',
                ),
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                ('human', '{input}'),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )

        agent = create_tool_calling_agent(self._llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )

    @staticmethod
    def _build_chat_history(history: list[ChatTurn]) -> list[HumanMessage | AIMessage]:
        messages: list[HumanMessage | AIMessage] = []
        for turn in history:
            if turn.role == 'assistant':
                messages.append(AIMessage(content=turn.content))
            else:
                messages.append(HumanMessage(content=turn.content))
        return messages

    @staticmethod
    def _build_document_filter(selected_documents: list[str]) -> Callable[[dict[str, Any]], bool]:
        selected = set(selected_documents)

        def metadata_filter(metadata: dict[str, Any]) -> bool:
            return metadata.get('source') in selected

        return metadata_filter

    def _index_exists(self) -> bool:
        return (
            self.settings.vectorstore_dir.joinpath('index.faiss').exists()
            and self.settings.vectorstore_dir.joinpath('index.pkl').exists()
        )

    @staticmethod
    def _to_source_document(doc: Document) -> SourceDocument:
        snippet = ' '.join(doc.page_content.split())[:240]
        page = doc.metadata.get('page')
        return SourceDocument(
            source=doc.metadata.get('source', 'unknown'),
            page=page + 1 if isinstance(page, int) else None,
            snippet=snippet,
        )


@lru_cache
def get_qa_service() -> QAService:
    return QAService(get_settings())
