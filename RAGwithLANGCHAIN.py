# pip install langchain chromadb openai

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Step 1: Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 2: Store documents in Chroma (vector DB)
texts = [
    "King is a male ruler of a country.",
    "Queen is a female ruler of a country.",
    "Man is an adult male human.",
    "Woman is an adult female human."
]

vectorstore = Chroma.from_texts(texts, embeddings)

# Step 3: Create RetrievalQA chain (RAG)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever
)

# Step 4: Ask a question
query = "Who is a female ruler?"
answer = qa.run(query)
print("RAG Answer (LangChain):", answer)
