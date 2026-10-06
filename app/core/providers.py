from langchain_openai import ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def validate_provider(settings):
    if settings.ai_provider == 'groq':
        if not settings.groq_api_key:
            raise EnvironmentError('The demo owner needs to configure the AI service.')
    elif settings.ai_provider == 'azure':
        if not (settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_chat_deployment):
            raise EnvironmentError('Azure chat configuration is incomplete.')
    else:
        raise EnvironmentError('AI_PROVIDER must be groq or azure.')
    if settings.embedding_provider not in {'fastembed', 'azure'}:
        raise EnvironmentError('EMBEDDING_PROVIDER must be fastembed or azure.')
    if settings.embedding_provider == 'azure' and not (settings.azure_openai_embedding_deployment and settings.azure_openai_api_key and settings.azure_openai_endpoint):
        raise EnvironmentError('Azure embedding configuration is incomplete.')


def make_llm(settings):
    validate_provider(settings)
    if settings.ai_provider == 'groq':
        extra = {'reasoning_effort': 'low'} if settings.groq_model.startswith('openai/gpt-oss') else {}
        return ChatOpenAI(model=settings.groq_model, api_key=settings.groq_api_key,
                          base_url='https://api.groq.com/openai/v1', max_tokens=700,
                          temperature=1, timeout=30, max_retries=0, extra_body=extra)
    return AzureChatOpenAI(azure_deployment=settings.azure_openai_chat_deployment,
                          model=settings.azure_openai_chat_model,
                          api_version=settings.azure_openai_api_version,
                          azure_endpoint=settings.azure_openai_endpoint,
                          api_key=settings.azure_openai_api_key,
                          temperature=settings.temperature, max_tokens=700,
                          timeout=30, max_retries=0)


def make_embeddings(settings):
    if settings.embedding_provider == 'fastembed':
        return FastEmbedEmbeddings(model_name=settings.embedding_model, threads=2, batch_size=16)
    return AzureOpenAIEmbeddings(model=settings.azure_openai_embedding_model or settings.azure_openai_embedding_deployment,
                                azure_deployment=settings.azure_openai_embedding_deployment,
                                api_version=settings.azure_openai_api_version,
                                azure_endpoint=settings.azure_openai_endpoint,
                                api_key=settings.azure_openai_api_key, max_retries=0,
                                request_timeout=30)
