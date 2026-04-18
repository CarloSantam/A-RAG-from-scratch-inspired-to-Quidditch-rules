from rag_begin import *
import streamlit as st
import os
import json
import uuid

# =========================
# Page configuration
# =========================

model="text-embedding-3-small"

st.set_page_config(
    page_title="A Rag Inspired to Quidditch",
    page_icon="🧹",
    layout="centered"
)

pwd = st.text_input("Inserisci la tua OPEN AI KEY:",type="password")


# =========================
# Load data and embeddings
# =========================

@st.cache_resource
def load_data_and_emb(pwd=''):
    """
    Load the knowledge base and its embeddings.
    If embeddings are not found, compute and save them.
    """
    json_path = "quidditch_regolamento.json"
    emb_path = "embeddings_db"

    # Check that the JSON file exists
    if not os.path.exists(json_path):
        st.error("⚠️ File `Quidditch_regolamento.json` non trovato nel repository.")
        st.stop()

    # Load JSON content
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract sentence list from JSON
    try:
        frasi = [item["sentence"] for item in data]
    except (TypeError, KeyError):
        st.error("⚠️ Il file JSON non ha il formato atteso. Ogni elemento deve contenere la chiave `sentence`.")
        st.stop()

    if not frasi:
        st.error("⚠️ Il file JSON è vuoto o non contiene frasi valide.")
        st.stop()

    # Load precomputed embeddings if available, otherwise create them
    if os.path.exists(f"{emb_path}.npy"):
        embeddings = load_emb(emb_path)
    else:
        embeddings = index_database(frasi, emb_path, model,pwd)

    return frasi, embeddings


frasi, embeddings = load_data_and_emb(pwd)


# =========================
# App UI
# =========================

st.markdown(
    """
    <div style="font-size: 2em; font-weight: bold; display: flex; align-items: center;">
        A Rag Inspired to Quidditch
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Fai una domanda sul Quidditch e ricevi una risposta basata sul contesto.")

st.write(
    "Questa web app è **non ufficiale** ed è solo un esperimento dell'autore. "
)


if pwd:
    
    # User input
    query = st.text_input("Inserisci la tua domanda:")
    
    # OpenAI model name
    model_llm_name = "gpt-5.1"
    
    
    # =========================
    # Run QA pipeline
    # =========================
    
    if query:
        with st.spinner("Sto pensando..."):
            try:
                # Generate answer, retrieved context, and ranked chunks
                answer, context, retrieved,query_emb,embeddings = Quidditch_gpt_core(
                    query=query,
                    frasi=frasi,
                    embeddings=embeddings,
                    model_emb=model,
                    model_llm_name=model_llm_name,
                    top_k=5,
                    min_similarity=0.2,
                    pwd=pwd
                )
    
                # Show answer
                st.subheader("Risposta")
                st.markdown(answer)
    
                # Optional debug / transparency section
                with st.expander("Contesto recuperato"):
                    st.text(context)
    
                with st.expander("Frasi più rilevanti"):
                    for i, (sentence, score) in enumerate(retrieved, start=1):
                        st.write(f"{i}. **Score:** {score:.3f}")
                        st.write(sentence)
                
                with st.expander("Grafico"):
                    fig=viz(query_emb,embeddings,query,retrieved)
                    
                    st.plotly_chart(fig,key=str(uuid.uuid4()))
    
            except Exception as e:
                st.error(f"❌ Errore nell'esecuzione del modello: {e}")
