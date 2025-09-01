import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
import torch

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
COLLECTION_NAME = "idearq-e5-large-instruct"
QDRANT_URL = "http://localhost:6333"

class E5InstructEmbeddings(Embeddings):
    def __init__(self, model_name=MODEL_NAME, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    def embed_documents(self, texts):
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed_texts, device=self.device).tolist()
    def embed_query(self, text):
        prefixed_text = f"query: {text}"
        return self.model.encode([prefixed_text], device=self.device)[0].tolist()

if __name__ == "__main__":
    load_dotenv()
    embeddings = E5InstructEmbeddings()
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
