from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def discover_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.settings.documents_dir.glob('*.pdf')
            if path.is_file()
        )

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []

        for path in self.discover_files():
            loader = PyPDFLoader(str(path))
            loaded_docs = loader.load()

            for document in loaded_docs:
                document.metadata['source'] = path.name
                document.metadata['file_name'] = path.name

            documents.extend(loaded_docs)

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        return splitter.split_documents(documents)
