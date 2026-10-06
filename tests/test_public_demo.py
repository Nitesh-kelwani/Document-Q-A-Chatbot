from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import pytest
from pypdf import PdfReader, PdfWriter
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from app.core.config import Settings
from app.core.providers import validate_provider, make_llm
from app.demo_runtime import UsageLimit, safe_error
from app.services.public_demo import PublicQAService, VisitorRegistry, parse_pdf, MAX_UPLOAD_BYTES, TTL_SECONDS

SAMPLES = Path(__file__).parents[1] / 'samples'


@pytest.fixture
def qa():
    registry = VisitorRegistry(start_cleanup=False)
    return PublicQAService(DeterministicFakeEmbedding(size=16), None, SAMPLES, registry)


def pdf_bytes():
    return (SAMPLES / 'company-handbook.pdf').read_bytes()


def test_samples_have_text_and_page_numbers():
    docs = parse_pdf('sample.pdf', pdf_bytes())
    assert [doc.metadata['page'] for doc in docs] == [1, 2]
    assert 'three days' in docs[0].page_content
    assert '14 calendar days' in docs[1].page_content


def test_upload_limits_and_bad_files(qa):
    _, visitor = qa.registry.get()
    for name in ['one.pdf', 'two.pdf', 'three.pdf']:
        qa.upload(visitor, name, pdf_bytes())
    with pytest.raises(ValueError, match='three'):
        qa.upload(visitor, 'four.pdf', pdf_bytes())
    assert len(visitor.uploads) == 3
    with pytest.raises(ValueError, match='5 MB'):
        parse_pdf('large.pdf', b'%PDF-' + b'x' * MAX_UPLOAD_BYTES)
    for name, content in [('bad.pdf', b'not a pdf'), ('broken.pdf', b'%PDF-1.7\ninvalid'), ('../escape.pdf', pdf_bytes())]:
        with pytest.raises(ValueError):
            parse_pdf(name, content)
    writer = PdfWriter()
    for _ in range(31):
        writer.add_blank_page(width=600, height=800)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match='30 pages'):
        parse_pdf('long.pdf', stream.getvalue())
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match='OCR'):
        parse_pdf('scan.pdf', stream.getvalue())
    writer.encrypt('password')
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match='Encrypted'):
        parse_pdf('encrypted.pdf', stream.getvalue())


def test_concurrent_visitors_and_reset(qa):
    _, a = qa.registry.get()
    _, b = qa.registry.get()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args: qa.upload(*args, pdf_bytes()), [(a, 'alice.pdf'), (b, 'bob.pdf')]))
    assert 'alice.pdf' not in qa.list_documents(b)
    assert 'bob.pdf' not in qa.list_documents(a)
    assert a.index is not b.index
    with pytest.raises(ValueError, match='current session'):
        qa.answer(b, 'Show private information', ['alice.pdf'])
    a.history.append({'role': 'user', 'content': 'private question'})
    a.clear()
    assert a.uploads == {} and a.index is None and a.history == []
    assert 'bob.pdf' in qa.list_documents(b)


def test_expiry_erases_live_references():
    now = [0]
    registry = VisitorRegistry(clock=lambda: now[0], start_cleanup=False)
    token, visitor = registry.get()
    visitor.uploads['private.pdf'] = ['private contents']
    visitor.history.append({'role': 'user', 'content': 'private'})
    visitor.index = object()
    visitor.sources.append('private citation')
    now[0] = TTL_SECONDS
    registry.purge()
    assert token not in registry.visitors
    assert visitor.uploads == {} and visitor.index is None and visitor.history == [] and visitor.sources == []
    new_token, fresh = registry.get(token)
    assert new_token != token and fresh is not visitor


class ToolChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_uses_scoped_sources(qa):
    qa.llm = ToolChatModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'search_selected_documents', 'args': {'query': 'remote work'}, 'id': 'call1', 'type': 'tool_call'}]),
        AIMessage(content='Up to three days per week (company-handbook.pdf, page 1).'),
    ])
    _, visitor = qa.registry.get()
    result = qa.answer(visitor, 'How many remote days?', ['company-handbook.pdf'])
    assert 'three days' in result.answer
    assert result.sources and all(source.source == 'company-handbook.pdf' for source in result.sources)
    assert all(source.page in {1, 2} for source in result.sources)
    assert len(visitor.history) == 2


def test_agent_cannot_answer_without_retrieval(qa):
    qa.llm = ToolChatModel(responses=[AIMessage(content='Invented salary is INR 900000')])
    _, visitor = qa.registry.get()
    result = qa.answer(visitor, 'What are employee salaries?', ['company-handbook.pdf'])
    assert "don't know" in result.answer and not result.sources


def test_agent_iteration_limit(qa):
    qa.llm = ToolChatModel(responses=[AIMessage(content='', tool_calls=[
        {'name': 'search_selected_documents', 'args': {'query': 'remote work'}, 'id': 'loop', 'type': 'tool_call'}])])
    _, visitor = qa.registry.get()
    with pytest.raises(TimeoutError):
        qa.answer(visitor, 'Loop forever please', ['company-handbook.pdf'])


def test_provider_setup_and_redacted_errors():
    with pytest.raises(EnvironmentError):
        validate_provider(Settings(_env_file=None, GROQ_API_KEY=''))
    settings = Settings(_env_file=None, GROQ_API_KEY='test-only-placeholder')
    llm = make_llm(settings)
    assert llm.max_tokens == 700 and llm.max_retries == 0
    assert 'private-key' not in safe_error(RuntimeError('private-key'))


def test_missing_secrets_ui(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'demo_app.py')).run(timeout=30)
    assert not app.exception
    assert any('not configured' in info.value for info in app.info)


def test_ready_ui_chat_and_reset(monkeypatch):
    import app.core.providers as providers
    monkeypatch.setattr(providers, 'validate_provider', lambda settings: None)
    monkeypatch.setattr(providers, 'make_embeddings', lambda settings: DeterministicFakeEmbedding(size=16))
    model = ToolChatModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'search_selected_documents', 'args': {'query': 'remote work'}, 'id': 'call1', 'type': 'tool_call'}]),
        AIMessage(content='Three days per week (company-handbook.pdf, page 1).'),
    ])
    monkeypatch.setattr(providers, 'make_llm', lambda settings: model)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'demo_app.py')).run(timeout=30)
    assert not app.exception
    assert sorted(app.multiselect[0].value) == ['company-handbook.pdf', 'product-guide.pdf']
    app.chat_input[0].set_value('How many remote work days?').run(timeout=30)
    assert not app.exception
    assert any('Three days' in item.value for item in app.markdown)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.chat_message
