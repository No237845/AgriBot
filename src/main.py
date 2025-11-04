

# Lancer avec : uvicorn src.api:app --reload



## =========================================================
# 🌾 AgriBot Burkina - API Principale
# =========================================================
# Gère la clé API, les crédits et communique avec le serveur RAG
# =========================================================
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import *




app = FastAPI(title="AgriBot Burkina API", version="2.0")

# Autoriser le frontend à appeler ton API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://127.0.0.1:3000"] si tu veux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Initialisation automatique du RAG au lancement de l’API"""
    global chain

    logging.info("🚀 Initialisation du pipeline RAG...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    corpus_docs = load_corpus(CORPUS_PATH)
    text_docs = load_text_source(SOURCE_PATH)
    all_docs = corpus_docs + text_docs

    vector_db = build_or_load_vector_db(all_docs, embeddings)
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=512)
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)
    logging.info("✅ RAG initialisé avec succès.")


# =====================================
# 5️ - ENDPOINTS API
# =====================================
class PromptRequest(BaseModel):
    prompt: str



@app.post("/generate")
def generate(request: PromptRequest):
    """Génère une réponse agricole à partir d'une question utilisateur"""
    global chain
    if not chain:
        raise HTTPException(status_code=500, detail="Pipeline RAG non initialisé.")

    try:
        # Passer la question sous forme de dictionnaire avec la clé attendue
        response = chain.invoke({"question": request.prompt})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de génération : {e}")



@app.get("/")
def home():
    """Endpoint de test simple"""
    return {"message": "Bienvenue sur AgriBot Burkina 🌾", "status": "OK"}