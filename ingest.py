import fitz
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text
    return full_text

def chunk_text(text: str, chunk_size: int =500, overlap: int = 50) ->list:
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunk = " ".join(chunk)
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_texts(chunks: list) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings = []
    for item in response.data:
        embeddings.append(item.embedding)
    return embeddings