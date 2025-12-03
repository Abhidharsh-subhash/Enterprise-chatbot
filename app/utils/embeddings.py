import openai
from app.core.config import settings

openai.api_key = settings.openai_api_key


def get_embedding(text: str):
    response = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding
