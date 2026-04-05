from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    DocumentListResponse,
    IngestionResponse,
)
from app.services.qa_service import QAService, get_qa_service


router = APIRouter(tags=['document-qa'])


@router.get('/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/documents', response_model=DocumentListResponse)
def list_documents(
    qa_service: QAService = Depends(get_qa_service),
) -> DocumentListResponse:
    return DocumentListResponse(documents=qa_service.list_documents())


@router.post('/documents/upload', response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...),
    qa_service: QAService = Depends(get_qa_service),
) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Uploaded file must include a file name.',
        )

    filename = Path(file.filename).name
    if Path(filename).suffix.lower() != '.pdf':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only PDF files are supported in this MVP.',
        )

    destination = qa_service.documents_dir / filename
    destination.write_bytes(await file.read())

    try:
        result = qa_service.reindex_corpus()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Indexing failed: {exc}',
        ) from exc

    return IngestionResponse(
        message=f"Uploaded '{filename}' and rebuilt the FAISS index.",
        files_indexed=result.files_indexed,
        chunks_indexed=result.chunks_indexed,
    )


@router.post('/documents/reindex', response_model=IngestionResponse)
def reindex_documents(
    qa_service: QAService = Depends(get_qa_service),
) -> IngestionResponse:
    try:
        result = qa_service.reindex_corpus()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Indexing failed: {exc}',
        ) from exc

    return IngestionResponse(
        message='Rebuilt the FAISS index from the PDF directory.',
        files_indexed=result.files_indexed,
        chunks_indexed=result.chunks_indexed,
    )


@router.post('/chat/ask', response_model=AnswerResponse)
def ask_question(
    payload: AnswerRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AnswerResponse:
    try:
        result = qa_service.answer_question(
            question=payload.question,
            history=payload.history,
            selected_documents=payload.selected_documents,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Chat request failed: {exc}',
        ) from exc

    return AnswerResponse(
        answer=result.answer,
        sources=result.sources,
        retrieved_chunks=result.retrieved_chunks,
    )
