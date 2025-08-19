# Step 1: Install ChromaDB
# pip install chromadb

# Step 2: Import libraries
import chromadb
from chromadb.utils import embedding_functions

# Step 3: Initialize ChromaDB client
client = chromadb.Client()

# Step 4: Create / Get a collection
collection = client.create_collection(name="my_collection")

# Step 5: Add documents with IDs
collection.add(
    documents=[
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Machine Learning is a subset of AI that learns from data.",
        "Deep Learning uses neural networks with many layers."
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Step 6: Query the collection
results = collection.query(
    query_texts=["What is AI?"],  # Always a list of queries
    n_results=2                   # How many results to return
)

print("Query Results:")
print(results)

# Step 7: Show embeddings manually (optional, for understanding)
default_embedding = embedding_functions.DefaultEmbeddingFunction()

embedding = default_embedding(["Artificial Intelligence is amazing!"])
print("\nGenerated Embedding (vector):")
print(embedding)
