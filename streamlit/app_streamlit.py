import streamlit as st
import requests
import time
import pandas as pd
import os

# Configuración de la API del Backend 
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")

# Funciones de la API 
def start_evaluation(settings: dict) -> str | None:
    try:
        response = requests.post(f"{BACKEND_URL}/evaluate", json=settings)
        response.raise_for_status()
        return response.json().get("job_id")
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con el backend: {e}")
        return None

def get_status(job_id: str) -> dict | None:
    try:
        response = requests.get(f"{BACKEND_URL}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al consultar el estado: {e}")
        return None

# Interfaz de Streamlit 
st.set_page_config(layout="wide", page_title="RAG-IDEArq")
st.title("🏛️ Sistema RAG de IDEArq")
# st.markdown("Herramienta para ejecutar y evaluar múltiples combinaciones de LLMs, embeddings y prompts.")

# Barra Lateral de Configuración 
with st.sidebar:
    st.header("⚙️ Parámetros de ejecución")
    question = st.text_area("Haz una pregunta", "¿Cuáles son las dataciones más antiguas para la minería del sílex en el centro peninsular?", height=100)
    st.markdown("---")
    
    # Configuración fija
    selected_llms = ['Phi-3-mini-4k-instruct']
    selected_embeddings = ["e5-large-instruct"]

    # Solo permitir selección de prompts
    prompt_options = ['prompt_one_shot', 'prompt_few_shot']
    selected_prompts = st.selectbox("Seleccionar prompt", options=prompt_options)

    run_button = st.button("🚀 Ejecutar", use_container_width=True)

# Lógica de Estado de la Sesión 
if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if run_button:
    if not all([question, selected_prompts]):
        st.error("Por favor, rellena todos los campos.")
    else:
        settings = {
            "question": question, "selected_llms": selected_llms,
            "selected_embeddings": selected_embeddings, "selected_prompts": [selected_prompts],
            "retriever_k": 10
        }
        job_id = start_evaluation(settings)
        if job_id:
            st.session_state.job_id = job_id
            st.info(f"Trabajo iniciado con ID: {job_id}. La interfaz se actualizará automáticamente.")

# Resultados
if st.session_state.job_id:
    job_id = st.session_state.job_id
    status_text = st.empty()
    results_container = st.container()

    while True:
        status_data = get_status(job_id)
        if not status_data:
            st.error("No se pudo obtener el estado. Deteniendo.")
            break

        status = status_data.get("status")
        if status == "running":
            status_text.info("🔄 Generando respuesta...")
            time.sleep(2)

        elif status == "complete":
            status_text.success("✅ ¡Respuesta generada!")
            all_results = status_data.get("results", [])

            if all_results:
                result = all_results[0] 

                with results_container:
                    # Mostrar información del modelo
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**🤖 LLM:** {result.get('llm', 'N/A')}")
                    with col2:
                        st.info(f"**🔍 Embedding:** {result.get('embedding', 'N/A')}")

                    # Mostrar la respuesta
                    st.markdown("### 💬 Respuesta:")
                    st.write(result.get('answer', 'Sin respuesta'))

                    # Mostrar información adicional 
                    with st.expander("📊 Detalles técnicos"):
                        st.write(f"**Prompt usado:** {result.get('prompt', 'N/A')}")
                        st.write(f"**Tiempo de respuesta:** {result.get('latency', 0):.2f} segundos")
                        st.write(f"**Documentos consultados:** {len(result.get('docs', []))}")

            st.session_state.job_id = None
            break

        elif "error" in status_data:
            st.error(f"❌ Error: {status_data['error']}")
            st.session_state.job_id = None
            break
