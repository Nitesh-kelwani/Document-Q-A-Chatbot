"""Community Cloud entrypoint; the FastAPI server is not needed."""
from hashlib import sha256
from pathlib import Path
from threading import Lock
import streamlit as st
from app.core.config import Settings
from app.core.providers import validate_provider, make_embeddings, make_llm
from app.demo_runtime import safe_error, InputError, UsageLimit
from app.services.public_demo import PublicQAService

st.set_page_config(page_title='Document Q&A | Nitesh Kelwani', page_icon='📄', layout='wide')
st.title('Document Q&A')
st.caption('Explore sample PDFs or upload your own. Answers include source passages and page references.')


@st.cache_resource
def service():
    settings = Settings()
    validate_provider(settings)
    return PublicQAService(make_embeddings(settings), make_llm(settings), Path(__file__).parent / 'samples')


@st.cache_resource
def request_gate():
    return Lock()


try:
    qa = service()
    previous_token = st.session_state.get('visitor_token')
    token, visitor = qa.registry.get(previous_token)
    if token != previous_token:
        st.session_state.visitor_token = token
        st.session_state.upload_signatures = {}
        st.session_state.selection = qa.list_documents(visitor)
        st.session_state.upload_generation = st.session_state.get('upload_generation', 0) + 1
        if previous_token:
            st.info('Your previous session expired. Uploaded files and chat were cleared.')
except EnvironmentError:
    st.info("This demo's AI service is not configured yet. Please check back soon.")
    st.stop()
except Exception as exc:
    st.warning(safe_error(exc))
    st.stop()

with st.sidebar:
    st.subheader('Your documents')
    st.caption('Samples are synthetic. Upload up to three text-based PDFs, each up to 5 MB and 30 pages.')
    st.caption('Uploads and indexes stay in this server session and are cleared on reset or after 30 minutes of inactivity. '
               'Questions and relevant PDF excerpts are sent to the AI provider. Avoid sensitive documents.')
    if st.button('Reset documents and chat', use_container_width=True):
        visitor.clear()
        st.session_state.upload_signatures = {}
        st.session_state.selection = qa.list_documents(visitor)
        st.session_state.upload_generation += 1
        st.rerun()
    files = st.file_uploader('Upload PDFs', type=['pdf'], accept_multiple_files=True,
                            key=f"upload-{st.session_state.upload_generation}")
    indexed_upload = False
    for uploaded in files:
        signature = sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.upload_signatures.get(uploaded.name) == signature:
            continue
        gate = request_gate()
        if not gate.acquire(blocking=False):
            st.warning('The demo is busy. Please try the upload again shortly.')
            break
        try:
            with st.spinner('Reading and indexing your PDF...'):
                qa.upload(visitor, uploaded.name, uploaded.getvalue())
            st.session_state.upload_signatures[uploaded.name] = signature
            if uploaded.name not in st.session_state.selection:
                st.session_state.selection.append(uploaded.name)
            st.success(f'Indexed {uploaded.name}')
            indexed_upload = True
        except (InputError, UsageLimit) as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.warning(safe_error(exc))
        finally:
            gate.release()
    if indexed_upload:
        # Release the file-upload widget's raw bytes after successful indexing.
        st.session_state.upload_generation += 1
        st.rerun()
    options = qa.list_documents(visitor)
    st.session_state.selection = [name for name in st.session_state.selection if name in options]
    selected = st.multiselect('Search these PDFs', options, key='selection')
    st.caption('Free demo · five questions per minute · shared AI quotas apply')

if not visitor.history:
    st.markdown('**Try the samples:**\n- How many remote work days are allowed?\n- What is the expense reimbursement deadline?\n- How do I export a report from Atlas?')
    with st.expander('Download sample PDFs'):
        for sample in sorted((Path(__file__).parent / 'samples').glob('*.pdf')):
            st.download_button(sample.name, sample.read_bytes(), sample.name, 'application/pdf')

for item in visitor.history:
    with st.chat_message(item['role']):
        st.markdown(item['content'])
if visitor.sources:
    with st.expander('Sources for the latest answer', expanded=True):
        for source in visitor.sources:
            st.caption(f'{source.source}, page {source.page}: {source.snippet}')

question = st.chat_input('Ask about the selected PDFs', disabled=not selected, max_chars=1000)
if question:
    gate = request_gate()
    if not gate.acquire(blocking=False):
        st.warning('The demo is serving another request. Please try again shortly.')
    else:
        try:
            with st.spinner('Searching your selected documents...'):
                result = qa.answer(visitor, question, selected)
        except (InputError, UsageLimit) as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.warning(safe_error(exc))
        else:
            st.rerun()
        finally:
            gate.release()
