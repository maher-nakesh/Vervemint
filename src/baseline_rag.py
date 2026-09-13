from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load documents
documents = []

folder = Path("data")

for file in folder.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append(text)


# 2. Split documents into chunks
def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


chunks = []

for document in documents:
    chunks.extend(chunk_text(document))


# 3. Create embeddings
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embeddings = embedding_model.encode(chunks)


# 4. User question
query = input("Ask a question: ")


# 5. Embed the question
query_embedding = embedding_model.encode([query])


# 6. Find similar chunks
scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

best_indexes = scores.argsort()[::-1]


# 7. Get top 3 chunks
context = "\n\n".join(
    chunks[index]
    for index in best_indexes[:3]
)


# 8. Ask Ollama
prompt = f"""Context:
{context}

Question: {query}

Answer:"""


response = ollama.chat(
    model="qwen2.5-7b-gpu:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# 9. Print answer
print("\nAnswer:")
print(response["message"]["content"])
