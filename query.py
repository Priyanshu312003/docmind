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
        n_results=min(top_k, collection.count()),
    )
    return results["documents"][0]

def build_prompt(chunks: list, question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
        f"If the answer is partially in the context, use what is available. Only say 'I don't have enough information' if there is truly nothing relevant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def ask(question: str) -> str:
    embed_question = embed_query(question)
    chunks = retrieve_chunks(embed_question)
    prompt = build_prompt(chunks, question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\nAnswer: {answer}")
    return answer

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if(question.lower() == "quit"):
            break
        ask(question)