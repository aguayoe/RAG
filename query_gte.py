import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
import torch

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
COLLECTION_NAME = "idearq-gte"
QDRANT_URL = "http://localhost:6333"

if __name__ == "__main__":
    load_dotenv()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device, 'trust_remote_code': True}
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=model_kwargs)
    client = QdrantClient(url=QDRANT_URL)
    vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    question = input("Introduce tu pregunta: ")
    retrieved_docs = vector_store.similarity_search(question, k=4)
    print("\nArtículos recuperados:")
    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        titulo = meta.get('title', 'Sin título')
        cita = meta.get('citation', 'Sin cita')
        autores = meta.get('authors', 'Sin autores')
        año = meta.get('year', 'Sin año')
        print(f"\n--- Artículo {i+1} ---")
        print(f"Título: {titulo}")
        print(f"Cita: {cita}")
        print(f"Autores: {autores}")
        print(f"Año: {año}")
