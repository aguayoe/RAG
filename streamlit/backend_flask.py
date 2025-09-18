#!/usr/bin/env python3
"""
RAG Backend Flask Application

A Flask application for running RAG (Retrieval-Augmented Generation) evaluations
with multiple embedding models, LLMs, and prompts.

Author: Consolidated version
Date: 2025
"""

from __future__ import annotations

import os
import pathlib
import torch
import logging
import sys
import uuid
import time
import itertools
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import weaviate
from langchain_weaviate import WeaviateVectorStore
from datasets import Dataset
from ragas import evaluate
from ragas import metrics

# Load environment variables
load_dotenv(override=True)

# Configuration
class Config:
    """Application configuration settings."""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LangSmith configuration
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'RAG-IDEArq-Evaluation')
    
    # Weaviate configuration
    WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY', None)
    
    
    # File paths
    RESULTS_DIR = pathlib.Path('./results')
    
    # Flask configuration
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5001
    FLASK_DEBUG = False

    # Weaviate collections mapping
    WEAVIATE_COLLECTIONS = {
        'all-MiniLM-L6-v2': 'IdearqAllMiniLM',
        'e5-large-instruct': 'IdearqE5',
        'gte-multilingual-base': 'IdearqGTE',
    }
    
    # Prompt templates
    PROMPTS = {
        'prompt_one_shot': """Eres un asistente experto en arqueología, historia y datos geoespaciales. Responde la pregunta usando los contextos proporcionados, que pueden estar en español, inglés, francés, catalán o portugués.
Sintetiza información de todos los contextos relevantes independientemente de su idioma. Si encuentras información relevante en cualquier idioma, úsala para construir tu respuesta en el mismo idioma en el que se realiza la pregunta.

Contexto: {context}

Pregunta: {question}

Ejemplo:
Q: ¿Cuál es la utilidad de los análisis de isótopos de estroncio en Arqueología?
A: El tema del desplazamiento, la movilidad y la migración ha sido altamente destacado como uno de los cinco grandes retos de la investigación arqueológica contemporánea (Kintigh et al., 2014: 12). El uso del análisis de isótopos de estroncio es hoy en día uno de los métodos más eficaces para afrontar este reto, ofreciendo un enfoque sistemático, cuantitativo y comparable a la movilidad de las poblaciones humanas y animales del pasado (Larsen 2018).

Respuesta:
""",

        'prompt_few_shot': """Eres un asistente experto en arqueología, historia y datos geoespaciales. Responde la pregunta usando los contextos proporcionados, que pueden estar en español, inglés, francés, catalán o portugués.
Sintetiza información de todos los contextos relevantes independientemente de su idioma. Si encuentras información relevante en cualquier idioma, úsala para construir tu respuesta en el mismo idioma en el que se realiza la pregunta.

Contexto: {context}

Pregunta: {question}

Ejemplos:
Q: ¿Cuál es la utilidad de los análisis de isótopos de estroncio en Arqueología?
A: El análisis de isótopos de estroncio es uno de los métodos más eficaces para estudiar la movilidad de poblaciones humanas y animales del pasado, ofreciendo un enfoque sistemático y cuantitativo.

Q: ¿Qué técnicas de datación se usan en arqueología?
A: Las principales técnicas incluyen radiocarbono (C14), termoluminiscencia, dendrocronología y datación por potasio-argón, cada una apropiada para diferentes tipos de materiales y rangos temporales.

Q: ¿Cómo se estudian los patrones de asentamiento prehistóricos?
A: Se utilizan métodos como prospección arqueológica, análisis espacial con SIG, teledetección y excavaciones estratégicas para entender la distribución y organización de los sitios.

Respuesta:
"""
    }
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Set environment variables for LangChain tracing
        os.environ['LANGCHAIN_TRACING_V2'] = "true" if self.LANGSMITH_API_KEY else "false"

# Global configuration instance
config = Config()

# Logging configuration
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up application logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {level}")

# Custom embeddings implementation
class E5InstructEmbeddings(Embeddings):
    """Custom embedding class for E5 Instruct models with optimized memory management."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", device: str = None):
        """
        Initialize E5 Instruct embeddings.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = model_name
        
        # Force CPU for large models with limited GPU memory
        if "large" in model_name and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 8e9:  # Less than 8GB VRAM
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with the passage prefix.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed_texts, device=self.device).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with the query prefix.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        prefixed_text = f"query: {text}"
        return self.model.encode([prefixed_text], device=self.device)[0].tolist()

# Model management
class ModelManager:
    """Singleton class for managing all ML models and clients."""
    
    _instance = None
    
    def __new__(cls) -> 'ModelManager':
        """Ensure singleton pattern for model loading."""
        if cls._instance is None:
            logging.info("Initializing ModelManager - loading all models...")
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize_models()
            logging.info("All models loaded successfully.")
        return cls._instance
    
    def _initialize_models(self) -> None:
        """Initialize all required models and clients."""
        try:
            self._initialize_weaviate_client()
            self._initialize_embedding_models()
            self._initialize_llm_models()
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise
    
    def _initialize_weaviate_client(self) -> None:
        """Initialize Weaviate vector database client."""
        try:
            # Use Weaviate v4 client
            if config.WEAVIATE_API_KEY:
                self.weaviate_client = weaviate.connect_to_local(
                    host=config.WEAVIATE_URL.replace('http://', '').replace('https://', ''),
                    headers={"X-Weaviate-Api-Key": config.WEAVIATE_API_KEY}
                )
            else:
                # Extract host from URL for v4 client
                host = config.WEAVIATE_URL.replace('http://', '').replace('https://', '')
                if ':' in host:
                    host_part, port_part = host.split(':')
                    port = int(port_part)
                else:
                    host_part = host
                    port = 8080

                self.weaviate_client = weaviate.connect_to_local(
                    host=host_part,
                    port=port
                )
            logging.info(f"Weaviate client connected to {config.WEAVIATE_URL}")
        except Exception as e:
            logging.error(f"Failed to initialize Weaviate client: {e}")
            raise
    
    
    def _initialize_embedding_models(self) -> None:
        """Initialize all embedding models."""
        self.embedding_models = {}
        
        try:
            # All-MiniLM-L6-v2 model
            self.embedding_models["all-MiniLM-L6-v2"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": config.DEVICE},
                encode_kwargs={"device": config.DEVICE}
            )
            logging.info("Loaded all-MiniLM-L6-v2 embedding model")
            
            # E5 Large Instruct model (using CPU for stability)
            self.embedding_models["e5-large-instruct"] = E5InstructEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct",
                device="cpu"
            )
            logging.info("Loaded e5-large-instruct embedding model")
            
            # GTE Multilingual Base model
            self.embedding_models["gte-multilingual-base"] = HuggingFaceEmbeddings(
                model_name="Alibaba-NLP/gte-multilingual-base",
                model_kwargs={'device': config.DEVICE, 'trust_remote_code': True},
                encode_kwargs={"device": config.DEVICE}
            )
            logging.info("Loaded gte-multilingual-base embedding model")
            
        except Exception as e:
            logging.error(f"Failed to initialize embedding models: {e}")
            raise
    
    def _initialize_llm_models(self) -> None:
        """Initialize all LLM models."""
        self.llms = {}
        
        try:
            self.llms['Llama-3.2-3B-Instruct'] = Ollama(
                model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"
            )
            logging.info("Loaded Llama-3.2-3B-Instruct model")
            
            self.llms['Phi-3-mini-4k-instruct'] = Ollama(
                model="hf.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF:Q6_K"
            )
            logging.info("Loaded Phi-3-mini-4k-instruct model")
            
            self.llms['Qwen3-4B-Instruct-2507'] = Ollama(
                model="hopephoto/Qwen3-4B-Instruct-2507_q8"
            )
            logging.info("Loaded Qwen3-4B-Instruct-2507 model")
            
        except Exception as e:
            logging.error(f"Failed to initialize LLM models: {e}")
            raise
    
    def get_embedding_model(self, model_name: str):
        """Get a specific embedding model by name."""
        if model_name not in self.embedding_models:
            raise ValueError(f"Embedding model '{model_name}' not found. Available: {list(self.embedding_models.keys())}")
        return self.embedding_models[model_name]
    
    def get_llm_model(self, model_name: str):
        """Get a specific LLM model by name."""
        if model_name not in self.llms:
            raise ValueError(f"LLM model '{model_name}' not found. Available: {list(self.llms.keys())}")
        return self.llms[model_name]
    
    def list_available_models(self) -> Dict[str, Any]:
        """Return a dictionary of all available models."""
        return {
            "embedding_models": list(self.embedding_models.keys()),
            "llm_models": list(self.llms.keys())
        }

# Global model manager instance
model_manager = ModelManager()

# Data classes and evaluation utilities
@dataclass
class RunResult:
    """Data class for storing evaluation run results."""
    llm: str
    embedding: str
    prompt: str
    answer: Optional[str]
    docs: List[Dict[str, Any]]
    latency: float
    metadata: Dict[str, Any]

class RAGEvaluator:
    """Main class for handling RAG evaluation pipeline."""
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        self.model_manager = model_manager
    
    def get_vectorstore_for_embedding(self, embedding_name: str) -> WeaviateVectorStore:
        """
        Get a Weaviate vector store configured for a specific embedding model.

        Args:
            embedding_name: Name of the embedding model

        Returns:
            Configured Weaviate instance
        """
        collection_name = config.WEAVIATE_COLLECTIONS.get(embedding_name)
        if not collection_name:
            available_embeddings = list(config.WEAVIATE_COLLECTIONS.keys())
            raise ValueError(f"No collection found for embedding '{embedding_name}'. Available embeddings: {available_embeddings}")

        embedding_model = self.model_manager.get_embedding_model(embedding_name)

        # Verify collection exists in Weaviate
        try:
            # For v4 client, check if collection exists
            collections = self.model_manager.weaviate_client.collections.list_all()
            class_names = [col.name for col in collections]

            if collection_name not in class_names:
                logging.error(f"Collection '{collection_name}' not found in Weaviate. Available collections: {class_names}")
                raise ValueError(f"Collection '{collection_name}' does not exist in Weaviate database")

            logging.info(f"Using collection '{collection_name}' for embedding model '{embedding_name}'")

        except Exception as e:
            logging.error(f"Error checking Weaviate collections: {e}")
            # Continue anyway, let Weaviate handle the error

        return WeaviateVectorStore(
            client=self.model_manager.weaviate_client,
            index_name=collection_name,
            text_key="content",
            embedding=embedding_model,
            attributes=["filename", "title", "source", "chunk_index", "doc_index"]
        )
    
    
    def run_single_combo(self, params: Dict[str, Any]) -> RunResult:
        """
        Run a single RAG evaluation combination.
        
        Args:
            params: Dictionary containing evaluation parameters
            
        Returns:
            RunResult with evaluation results
        """
        start_time = time.time()
        
        try:
            # Document retrieval
            vectorstore = self.get_vectorstore_for_embedding(params['embedding_name'])
            
            # Configure retriever with proper parameters
            retriever_k = params.get('retriever_k', 10)  # Default to 10 if not specified
            search_type = params.get('search_type', 'similarity')  # Can be 'similarity' or 'mmr'
            
            if search_type == 'mmr':
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": retriever_k,
                        "fetch_k": retriever_k * 2,  # Fetch more for diversity
                        "lambda_mult": 0.7  # Balance between relevance and diversity
                    }
                )
            else:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": retriever_k}
                )
            
            docs = retriever.get_relevant_documents(params['question'])

            # Log retrieval results for debugging
            logging.info(f"Retrieved {len(docs)} documents for embedding '{params['embedding_name']}' from collection '{config.WEAVIATE_COLLECTIONS.get(params['embedding_name'])}'")
            if docs:
                logging.debug(f"First document preview: {docs[0].page_content[:200]}...")
            else:
                logging.warning(f"No documents retrieved for question: {params['question'][:100]}...")

            retrieval_time = time.time()

            # Prepare context and prompt (no reranking)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt_name = params['prompt_name']

            if prompt_name == 'hub_rlm_rag':
                template = hub.pull(config.PROMPTS[prompt_name])
            else:
                template_str = config.PROMPTS.get(prompt_name)
                if not template_str:
                    raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
                class Template:
                    def __init__(self, template_str):
                        self.template_str = template_str

                    def format(self, **kwargs):
                        return self.template_str.format(**kwargs)

                template = Template(template_str)

            prompt_text = template.format(context=context, question=params['question'])

            # Generate response
            llm = self.model_manager.get_llm_model(params['llm_name'])
            response = llm.invoke(prompt_text, config={"metadata": {"combo_params": params}})
            generation_time = time.time()

            return RunResult(
                llm=params['llm_name'],
                embedding=params['embedding_name'],
                prompt=params['prompt_name'],
                answer=response,
                docs=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
                latency=generation_time - start_time,
                metadata={
                    "retrieval_latency": retrieval_time - start_time,
                    "generation_latency": generation_time - retrieval_time,
                    "success": True
                }
            )
            
        except Exception as e:
            logging.error(f"Error in evaluation combo: {e}")
            return RunResult(
                llm=params.get('llm_name', 'unknown'),
                embedding=params.get('embedding_name', 'unknown'),
                prompt=params.get('prompt_name', 'unknown'),
                answer=f"Error: {str(e)}",
                docs=[],
                latency=time.time() - start_time,
                metadata={"error": str(e), "success": False}
            )
    

# Global evaluator instance
rag_evaluator = RAGEvaluator()

# Job management
class JobManager:
    """Manages background evaluation jobs and their status."""
    
    def __init__(self):
        """Initialize job manager with empty job storage."""
        self.job_status: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, job_id: str, settings: Dict[str, Any]) -> None:
        """
        Create a new evaluation job.
        
        Args:
            job_id: Unique job identifier
            settings: Evaluation settings and parameters
        """
        combos = list(itertools.product(
            settings['selected_llms'], 
            settings['selected_embeddings'], 
            settings['selected_prompts']
        ))
        
        self.job_status[job_id] = {
            "status": "running",
            "progress": 0,
            "total": len(combos),
            "results": [],
            "start_time": time.time(),
            "error": None
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if not found
        """
        return self.job_status.get(job_id)
    
    def update_job_progress(self, job_id: str, progress: int, result: Any = None) -> None:
        """
        Update job progress and add result if provided.
        
        Args:
            job_id: Job identifier
            progress: Current progress count
            result: Optional result to add
        """
        if job_id in self.job_status:
            self.job_status[job_id]["progress"] = progress
            if result is not None:
                self.job_status[job_id]["results"].append(result)
    
    def set_job_status(self, job_id: str, status: str, **kwargs) -> None:
        """
        Set job status and optional additional fields.
        
        Args:
            job_id: Job identifier
            status: New status
            **kwargs: Additional fields to update
        """
        if job_id in self.job_status:
            self.job_status[job_id]["status"] = status
            self.job_status[job_id].update(kwargs)
    
    def run_evaluation_job(self, job_id: str, settings: Dict[str, Any]) -> None:
        """
        Run evaluation job in background thread.
        
        Args:
            job_id: Job identifier
            settings: Evaluation settings
        """
        try:
            combos = list(itertools.product(
                settings['selected_llms'], 
                settings['selected_embeddings'], 
                settings['selected_prompts']
            ))
            
            logging.info(f"Starting evaluation job {job_id} with {len(combos)} combinations")
            
            for i, (llm, emb, prompt) in enumerate(combos):
                combo_params = {
                    **settings, 
                    "llm_name": llm, 
                    "embedding_name": emb, 
                    "prompt_name": prompt
                }
                
                try:
                    result = rag_evaluator.run_single_combo(combo_params)
                    self.update_job_progress(job_id, i + 1, asdict(result))
                    logging.info(f"Job {job_id}: Completed combo {i+1}/{len(combos)}")
                    
                except Exception as e:
                    logging.error(f"Error in combo {i+1} for job {job_id}: {e}")
                    # Continue with other combinations even if one fails
                    error_result = {
                        "llm": llm,
                        "embedding": emb,
                        "prompt": prompt,
                        "answer": f"Error: {str(e)}",
                        "docs": [],
                        "latency": 0,
                        "metadata": {"error": str(e), "success": False}
                    }
                    self.update_job_progress(job_id, i + 1, error_result)

            # Complete job
            self.set_job_status(
                job_id,
                "complete",
                end_time=time.time()
            )
            
            logging.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logging.error(f"Critical error in evaluation job {job_id}: {e}")
            self.set_job_status(
                job_id, 
                "failed", 
                error=str(e),
                end_time=time.time()
            )

# Global job manager instance
job_manager = JobManager()

# Flask application
def create_app() -> Flask:
    """
    Create and configure Flask application.
    
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/evaluate", methods=['POST'])
    def start_evaluation_endpoint():
        """Start a new evaluation job."""
        try:
            settings = request.json
            if not settings:
                return jsonify({"error": "No settings provided"}), 400
            
            # Validate required fields
            required_fields = ['question', 'selected_llms', 'selected_embeddings', 'selected_prompts']
            missing_fields = [field for field in required_fields if field not in settings]
            
            if missing_fields:
                return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
            
            job_id = str(uuid.uuid4())
            job_manager.create_job(job_id, settings)
            
            # Start evaluation in background thread
            thread = threading.Thread(
                target=job_manager.run_evaluation_job, 
                args=(job_id, settings)
            )
            thread.daemon = True
            thread.start()
            
            logging.info(f"Started evaluation job {job_id}")
            return jsonify({"job_id": job_id})
            
        except Exception as e:
            logging.error(f"Error starting evaluation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/status/<job_id>", methods=['GET'])
    def get_status_endpoint(job_id: str):
        """Get the status of an evaluation job."""
        try:
            status = job_manager.get_job_status(job_id)
            if not status:
                return jsonify({"error": "Job not found"}), 404
            
            return jsonify(status)
            
        except Exception as e:
            logging.error(f"Error getting job status for {job_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "available_models": rag_evaluator.model_manager.list_available_models()
        })

    @app.route("/test_retrieval", methods=['POST'])
    def test_retrieval_endpoint():
        """Test document retrieval for debugging purposes."""
        try:
            data = request.json
            if not data or 'question' not in data or 'embedding_name' not in data:
                return jsonify({"error": "Missing required fields: 'question' and 'embedding_name'"}), 400
            
            question = data['question']
            embedding_name = data['embedding_name']
            k = data.get('k', 5)
            
            # Test retrieval
            vectorstore = rag_evaluator.get_vectorstore_for_embedding(embedding_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(question)
            
            # Format response
            retrieved_docs = []
            for i, doc in enumerate(docs):
                retrieved_docs.append({
                    "index": i,
                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                })
            
            return jsonify({
                "question": question,
                "embedding_name": embedding_name,
                "collection_name": config.WEAVIATE_COLLECTIONS.get(embedding_name),
                "retrieved_count": len(docs),
                "requested_k": k,
                "documents": retrieved_docs
            })
            
        except Exception as e:
            logging.error(f"Error testing retrieval: {e}")
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logging.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500
    
    return app

def main() -> None:
    """
    Main entry point for the RAG backend application.
    
    Sets up logging, initializes the Flask app, and starts the server.
    """
    # Configure logging
    setup_logging(level="INFO")
    
    # Create Flask application
    app = create_app()
    
    # Start the server
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

if __name__ == "__main__":
    main()