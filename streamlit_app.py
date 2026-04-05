import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api')


def check_backend() -> tuple[bool, str]:
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        response.raise_for_status()
        return True, 'Backend connected'
    except requests.RequestException as exc:
        return False, f'Backend unavailable: {exc}'


def fetch_documents() -> tuple[list[str], str | None]:
    try:
        response = requests.get(f'{API_BASE_URL}/documents', timeout=10)
        response.raise_for_status()
        return response.json().get('documents', []), None
    except requests.RequestException as exc:
        return [], _extract_error_detail(exc)


def upload_pdf(uploaded_file) -> tuple[bool, str]:
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
    try:
        response = requests.post(f'{API_BASE_URL}/documents/upload', files=files, timeout=120)
        data = response.json()
        response.raise_for_status()
        return True, data['message']
    except requests.RequestException as exc:
        detail = _extract_error_detail(exc)
        return False, detail


def ask_question(
    question: str,
    history: list[dict[str, str]],
    selected_documents: list[str],
) -> dict[str, Any]:
    payload = {
        'question': question,
        'history': history,
        'selected_documents': selected_documents,
    }
    response = requests.post(f'{API_BASE_URL}/chat/ask', json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def _extract_error_detail(exc: requests.RequestException) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    try:
        return response.json().get('detail', response.text)
    except ValueError:
        return response.text


st.set_page_config(page_title='Document Q&A Chatbot', layout='wide')
st.title('Document Q&A Chatbot')
st.caption('FastAPI + Streamlit + Azure OpenAI + LangChain Agents')

if 'history' not in st.session_state:
    st.session_state.history = []
if 'indexed_file_signature' not in st.session_state:
    st.session_state.indexed_file_signature = None
if 'upload_status' not in st.session_state:
    st.session_state.upload_status = None
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []

backend_ok, backend_message = check_backend()
available_documents: list[str] = []
documents_error: str | None = None
if backend_ok:
    available_documents, documents_error = fetch_documents()
    st.session_state.selected_documents = [
        document
        for document in st.session_state.selected_documents
        if document in available_documents
    ]

with st.sidebar:
    st.subheader('Status')
    st.write(backend_message)

    st.subheader('Upload PDF')
    uploaded_file = st.file_uploader('Choose a PDF', type=['pdf'])
    if uploaded_file is not None:
        file_signature = f'{uploaded_file.name}:{uploaded_file.size}'

        if st.session_state.indexed_file_signature != file_signature:
            if backend_ok:
                with st.spinner('Uploading and indexing PDF...'):
                    success, message = upload_pdf(uploaded_file)
                st.session_state.upload_status = (success, message)
                if success:
                    st.session_state.indexed_file_signature = file_signature
                    available_documents, documents_error = fetch_documents()
                    if uploaded_file.name in available_documents:
                        if uploaded_file.name not in st.session_state.selected_documents:
                            st.session_state.selected_documents.append(uploaded_file.name)
                else:
                    st.session_state.indexed_file_signature = None
            else:
                st.session_state.upload_status = (
                    False,
                    'Backend unavailable. Start the FastAPI backend before uploading.',
                )

        if st.button('Reindex current PDF', use_container_width=True):
            if backend_ok:
                with st.spinner('Re-uploading and rebuilding index...'):
                    success, message = upload_pdf(uploaded_file)
                st.session_state.upload_status = (success, message)
                if success:
                    st.session_state.indexed_file_signature = file_signature
                    available_documents, documents_error = fetch_documents()
                    if uploaded_file.name in available_documents:
                        if uploaded_file.name not in st.session_state.selected_documents:
                            st.session_state.selected_documents.append(uploaded_file.name)
            else:
                st.session_state.upload_status = (
                    False,
                    'Backend unavailable. Start the FastAPI backend before uploading.',
                )

    if st.session_state.upload_status is not None:
        success, message = st.session_state.upload_status
        if success:
            st.success(message)
        else:
            st.error(message)

    st.subheader('Search Scope')
    if documents_error:
        st.error(documents_error)
    selected_documents = st.multiselect(
        'Choose indexed PDFs to search',
        options=available_documents,
        default=st.session_state.selected_documents,
        key='selected_documents',
    )
    if selected_documents:
        st.caption(f"Searching {len(selected_documents)} document(s)")
    elif available_documents:
        st.caption('Select one or more indexed PDFs before asking a question.')
    else:
        st.caption('Upload a PDF to create the searchable document list.')

for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask a question about your selected PDFs')
if prompt:
    st.session_state.history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    if not backend_ok:
        error_message = 'Start the FastAPI backend before asking questions.'
        st.session_state.history.append({'role': 'assistant', 'content': error_message})
        with st.chat_message('assistant'):
            st.error(error_message)
    elif not st.session_state.selected_documents:
        error_message = 'Select at least one indexed PDF in the sidebar before asking a question.'
        st.session_state.history.append({'role': 'assistant', 'content': error_message})
        with st.chat_message('assistant'):
            st.error(error_message)
    else:
        with st.chat_message('assistant'):
            with st.spinner('Searching selected documents...'):
                try:
                    result = ask_question(
                        prompt,
                        st.session_state.history[:-1],
                        st.session_state.selected_documents,
                    )
                    answer = result['answer']
                    st.markdown(answer)
                    if result['sources']:
                        st.markdown('**Sources**')
                        for source in result['sources']:
                            page = source.get('page')
                            page_label = f" (page {page})" if page else ''
                            st.caption(f"{source['source']}{page_label}: {source['snippet']}")
                    st.session_state.history.append({'role': 'assistant', 'content': answer})
                except requests.RequestException as exc:
                    error_message = _extract_error_detail(exc)
                    st.error(error_message)
                    st.session_state.history.append({'role': 'assistant', 'content': error_message})
