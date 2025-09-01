import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from pathlib import Path

# Configuración
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
VECTOR_SIZE = 768
COLLECTION_NAME = "idearq-gte"
QDRANT_URL = "http://localhost:6333"
PDF_FOLDER = "pdf_articulos_idearq/"

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device, 'trust_remote_code': True}
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=model_kwargs)
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
    logging.info("Ingesta completada para GTE.")
