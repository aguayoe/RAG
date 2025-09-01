import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from pathlib import Path

# Configuración
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
VECTOR_SIZE = 1024
COLLECTION_NAME = "idearq-e5-large-instruct"
QDRANT_URL = "http://localhost:6333"
PDF_FOLDER = "pdf_articulos_idearq/"

class E5InstructEmbeddings(Embeddings):
    def __init__(self, model_name=MODEL_NAME, device=None):
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f'Modelo E5 cargado en dispositivo: {device}')
    def embed_documents(self, texts):
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed_texts, device=self.device).tolist()
    def embed_query(self, text):
        prefixed_text = f"query: {text}"
        return self.model.encode([prefixed_text], device=self.device)[0].tolist()

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = 'cuda' if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else 'cpu'
    embeddings = E5InstructEmbeddings(model_name=MODEL_NAME, device=device)
    client = QdrantClient(url=QDRANT_URL)
    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION_NAME)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    all_docs = []
    folder_path = Path(PDF_FOLDER)
    for pdf_file in folder_path.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_file), mode="single")
            docs_from_file = loader.load()
            all_docs.extend(docs_from_file)
        except Exception as e:
            logging.warning(f"No se pudo cargar el archivo '{pdf_file.name}'. Saltando. Error: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""])
    all_splits = text_splitter.split_documents(all_docs)
    vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    vector_store.add_documents(all_splits)
    logging.info("Ingesta completada para E5.")
