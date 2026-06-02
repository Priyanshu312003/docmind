eval_set = [
    {
        "question": "What is a RAG?",
        "expected": "Retrieval-Augmented Generation"
    },
    {
        "question": "What are the two phases of RAG?",
        "expected": "Ingestion, Offline, Retrieval, Generation"
    },
    {
        "question":"explain chunking",
        "expected":"process of splitting a long document into smaller pieces"
    },
    {
        "question":"what are chunking strategies?",
        "expected":"Fixed size, Sentence-aware, Recursive, Semantic"
    },
    {
        "question":"What is cosine similarity?",
        "expected":"cosine, angle, vectors"
    },
    {
        "question":"What is FAISS?",
        "expected":"in-memory library, speed-focused"
    },
    {
        "question":"What are the key RAG evaluation metrics?",
        "expected":"retrieval recall, answer faithfulness, answer relevance"
    },
    {
        "question":"What is the purpose of openai library?",
        "expected":"Embeddings and GPT API calls"
    },
    {
        "question":"What is Reranking?",
        "expected":"Reranking adds a second pass that scores each retrieved chunk specifically for relevance to the question"
    },
    {
        "question":"What is Minimal effective RAG prompt structure?",
        "expected":"system instruction, context, question, answer"
    },
]

from query import ask

correct = 0

for item in eval_set:
    answer = ask(item["question"])
    expected = item["expected"].lower()
    
    if any(word in answer.lower() for word in expected.split(",")):
        correct += 1
        print(f"✓ {item['question']}")
    else:
        print(f"✗ {item['question']}")
        print(f"  Expected: {item['expected']}")
        print(f"  Got: {answer[:150]}")

print(f"\nScore: {correct}/10")