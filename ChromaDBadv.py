# ================================
# STEP 1: Install required packages
# ================================
# !pip install chromadb openai

# ================================
# STEP 2: Import required modules
# ================================
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np

# ================================
# STEP 3: Setup OpenAI client for embeddings
# ================================
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def get_embedding(text):
    """Get vector embedding for a given text using OpenAI embeddings."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ================================
# STEP 4: Setup ChromaDB client with persistence
# ================================
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",   # enables persistent storage
        persist_directory="./my_chroma_db"  # folder to store DB files
    )
)

# ================================
# STEP 5: Create or get a collection
# ================================
collection_name = "documents"
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# ================================
# STEP 6: Add documents (texts + embeddings + metadata)
# ================================
texts = [
    "Machine learning is amazing.",
    "ChromaDB is a vector database.",
    "RAG pipelines combine LLMs with vector search."
]
metadatas = [
    {"topic": "ML"},
    {"topic": "Vector DB"},
    {"topic": "RAG"}
]

# Convert texts to embeddings
embeddings = [get_embedding(t) for t in texts]

# Add documents to Chroma
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=["doc1", "doc2", "doc3"]
)

# Persist the database
collection.persist()

# ================================
# STEP 7: Query documents by similarity
# ================================
query_text = "Tell me about vector databases."
query_embedding = get_embedding(query_text)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,                  # number of similar documents
    include=["documents", "metadatas"]
)

print("Query results:")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"- {doc} | Metadata: {meta}")

# ================================
# STEP 8: Update a document
# ================================
collection.update(
    ids=["doc2"],
    documents=["ChromaDB stores vectors efficiently."],
    metadatas=[{"topic": "Vector DB Updated"}],
    embeddings=[get_embedding("ChromaDB stores vectors efficiently.")]
)

# ================================
# STEP 9: Delete a document
# ================================
collection.delete(ids=["doc1"])

# ================================
# STEP 10: Filter query using metadata
# ================================
query_embedding = get_embedding("I want to learn about RAG pipelines.")
filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"topic": "RAG"},
    include=["documents", "metadatas"]
)
print("Filtered query results:")
for doc, meta in zip(filtered_results['documents'][0], filtered_results['metadatas'][0]):
    print(f"- {doc} | Metadata: {meta}")

# ================================
# STEP 11: RAG Integration Example (Pseudo)
# ================================
# Step 1: User query
user_query = "Explain vector databases in simple terms."

# Step 2: Embed query
query_embedding = get_embedding(user_query)

# Step 3: Retrieve top relevant docs
retrieved_docs = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    include=["documents"]
)

# Step 4: Combine retrieved docs and feed into LLM
context = " ".join(retrieved_docs['documents'][0])
prompt = f"Answer this query based on context:\nContext: {context}\nQuery: {user_query}"

# Step 5: LLM answer (pseudo, can use GPT model)
# response = openai_client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role":"system","content":"You are an expert AI assistant."},
#               {"role":"user","content":prompt}]
# )
# print(response.choices[0].message.content)

