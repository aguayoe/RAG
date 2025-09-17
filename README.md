# RAG-CSIC

Repositorio para experimentos y pipelines de Retrieval-Augmented Generation (RAG) en Arqueología y Prehistoria.

## Estructura recomendada

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

## Recomendaciones
- Mantén los scripts de ingesta y consulta separados por modelo.
- Guarda los notebooks en la carpeta `notebooks/`.
- Los datos fuente (PDFs) deben ir en `pdf_articulos_idearq/`.
- Los resultados y análisis en `results/`.
- Documenta dependencias en `requirements.txt`.

## Uso rápido
1. Ejecuta el script de ingesta según el modelo de embedding.
2. Usa el script de consulta para hacer preguntas.
3. Realiza experimentos y análisis en los notebooks.

## Control de versiones
- Haz commits frecuentes y descriptivos.
- Sube el repositorio a GitHub para copia de seguridad y colaboración.
