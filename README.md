# RAG-IDEArq

Repositorio para experimentos y pipelines de un sistema Retrieval-Augmented Generation (RAG) sobre recursos de IDEArq (http://www.idearqueologia.org).

## Descripción

Este proyecto busca aplicar técnicas de RAG para enriquecer el análisis y la consulta de documentación arqueológica, facilitando la integración de modelos de lenguaje con una base de datos bibliográfica especializada.

![RAG Pipeline](img/Pipeline-RAG.drawio.png)
## Estructura
```
RAG/
├── rag-idearq-langgraph-weaviate.ipynb    # Notebook del RAG y evaluación
├── rag-indexacion-weaviate.ipynb          # Notebook de indexación                
├── results.pdf                            # Resultados de evaluaciones 
├── RAG-idearq/                            # Contenedor de docker de la base de datos Weaviate
|   ├──  weaviate_data                     # Base de datos vectorial              
|   └──  docker-compose.yml                # Fichero de instalación del contenedor
├── streamlit                              # Aplicación de Streamlit
|   ├── app_streamlit.py                   # Código de la aplicacióm
|   └── backend_flask.py                   # API de Flask
├── requirements.txt                       # Dependencias del proyecto
└── README.md                              # Documentación y guía
```

## Ejecución

El proyecto se ha realizado con Python 3.12.11 en un entorno de conda y se ha utilizado Weaviate como base de datos vectorial en un contenedor de Docker.
Los modelos de lenguaje se han descargado en local con Ollama y los modelos de embedding provienen de HuggingFace. Para realizar el RAG se han utilizado los frameworks LangChain y LangGraph, y se ha desarrollado una interfaz con Streamlit y Flask. Todas las API KEYS están almacenadas en un fichero .env que no se ha subido al repositorio.

```bash
git clone https://github.com/aguayoe/RAG.git
cd RAG
pip install -r requirements.txt
```
Para ejecutar streamlit, aplica cada línea de código en una terminal distinta:

```bash
python backend_flask.py
```
```bash
streamlit run app_streamlit.py
```
Y ejecutar el modelo de lenguaje en otra terminal (inicializando Ollama previamente):

```bash
ollama run hf.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF:Q6_K
```
Por último, levantar el contenedor de Docker:

```bash
docker compose up
```

## Licencia

CC0-1.0 license.

