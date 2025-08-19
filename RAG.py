# -------------------------------
# RAG Example with Chroma DB + OpenAI
# -------------------------------

# Step 1: Install required packages
# pip install chromadb openai

# Step 2: Import libraries
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Step 3: Initialize Chroma DB and OpenAI
client = chromadb.Client()

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_OPENAI_KEY",      # Replace with your OpenAI API key
    model_name="text-embedding-3-small"
)

# Create a collection to store documents
collection = client.create_collection(
    name="my_docs",
    embedding_function=ef
)

# OpenAI client for LLM generation
llm_client = OpenAI(api_key="YOUR_OPENAI_KEY")  # Replace with your OpenAI API key

# Step 4: Add documents to Chroma
docs = [
    {"id": "1", "text": "Elon Musk founded Tesla in 2003."},
    {"id": "2", "text": "Python is a popular programming language."},
    {"id": "3", "text": "The capital of France is Paris."}
]

for doc in docs:
    collection.add(
        documents=[doc['text']],
        ids=[doc['id']]
    )

# Step 5: Query the database (Retrieval)
query = "Who founded Tesla?"

results = collection.query(
    query_texts=[query],
    n_results=1
)

retrieved_doc = results['documents'][0][0]
print("Retrieved Document:", retrieved_doc)

# Step 6: Generate answer using LLM (Augmented Generation)
response = llm_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the question based on the following document:\n{retrieved_doc}\n\nQuestion: {query}"}
    ]
)

answer = response.choices[0].message["content"]
print("RAG Answer:", answer)

# -------------------------------
# Done! This is a full RAG workflow
# -------------------------------
