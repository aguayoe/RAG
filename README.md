# RAG-CSIC

Repositorio para experimentos y pipelines de un sistema Retrieval-Augmented Generation (RAG) sobre recursos de IDEArq.

## Descripción

Este proyecto busca aplicar técnicas de RAG para enriquecer el análisis y la consulta de documentación arqueológica, facilitando la integración de modelos de lenguaje con bases de datos especializadas.

## Estructura
```
RAG/
├── ingest_e5.py                # Script de ingesta para embeddings E5
├── ingest_gte.py               # Script de ingesta para embeddings GTE
├── query_e5.py                 # Script de consulta para embeddings E5
├── query_gte.py                # Script de consulta para embeddings GTE
├── notebooks/                  # Notebooks Jupyter para experimentos y análisis
│   ├── rag-tfm-e5-large.ipynb
│   ├── rag-tfm-gte.ipynb
│   ├── rag-idearq-langgraph-weaviate.ipynb
│   └── ...
├── pdf_articulos_idearq/       # Carpeta con PDFs fuente
├── results/                    # Resultados de evaluaciones y análisis
├── RAG-idearq/                 # Otros recursos, datos o configuraciones
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación y guía
```

## Instalación

Requiere Python 3.9+.

```bash
git clone https://github.com/aguayoe/RAG.git
cd RAG
pip install -r requirements.txt
```

## Uso rápido

**Ingesta de datos:**
```bash
python ingest_e5.py
```

**Consulta:**
```bash
python query_e5.py "¿Cuál es el contexto de la cueva X?"
```

**Experimentos:**
Abre los notebooks en la carpeta `notebooks/` con JupyterLab o VS Code.

## Contribuir

¿Ideas nuevas? ¡Bienvenidas!
- Abre un issue con sugerencias o errores.
- Haz un fork y envía tu Pull Request.
- Sigue el estilo de nombres y comentarios del repo.

## Licencia
CC0-1.0 license.
