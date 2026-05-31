from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def embed_query(query: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

def retrieve_chunks(query_embedding: list, top_k: int=5) -> list:
    chroma = chromadb.PersistentClient(path="./chroma_store")
    collection = chroma.get_collection(name="docmind")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(collection)),
    )
    return results["documents"][0]

def build_prompt(chunks: list, question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
        f"If the answer is not in the context, say 'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt