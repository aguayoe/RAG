# RAG-IDEArq

Repositorio para experimentos y pipelines de un sistema Retrieval-Augmented Generation (RAG) sobre recursos de IDEArq.

## Descripción

Este proyecto busca aplicar técnicas de RAG para enriquecer el análisis y la consulta de documentación arqueológica, facilitando la integración de modelos de lenguaje con bases de datos especializadas.

![RAG Pipeline](img/Pipeline-RAG.drawio.png)
## Estructura
```
RAG/
├── rag-idearq-langgraph-weaviate.ipynb    # Notebook del RAG y evaluación
├── rag-indexacion-weaviate.ipynb          # Notebook de indexación
├── rag-tfm-test.ipynb                     # Notebook de prueba
├── results/                               # Resultados de evaluaciones 
├── RAG-idearq/                            # Contenedor de docker de la base de datos Weaviate
├── requirements.txt                       # Dependencias del proyecto
└── README.md                              # Documentación y guía
```

## Instalación

El proyecto se ha realizado con Python 3.12.11 en un entorno de conda y se utilizado Weaviate como base de datos vectorial en un contenedor de Docker.

```bash
git clone https://github.com/aguayoe/RAG.git
cd RAG
pip install -r requirements.txt
```

## Licencia

CC0-1.0 license.

