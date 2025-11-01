# =========================================================
# AgriBot Burkina - Assistant IA agricole (RAG unifié)
# =========================================================

import os
import logging
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
#from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain.retrievers.multi_query import MultiQueryRetriever
#from langchain_community.retrievers import MultiQueryRetriever
#from langchain.retrievers.multi_query import MultiQueryRetrieverer
#from langchain_experimental.retrievers import MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever








import ollama

# =====================================
# CONFIGURATION
# =====================================
logging.basicConfig(level=logging.INFO)

PDF_PATH = "./data/"                # Dossier des fichiers PDF
CORPUS_PATH = "./data/corpus.json"  # Fichier JSON contenant les données collectées
SOURCE_PATH = "./data/source.txt"   # Fichier texte brut contenant des sources
VECTOR_DB_PATH = "./chrome_langchain_db"
VECTOR_COLLECTION = "agri-rag"

EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

# Initialiser le modèle d’embedding
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


# =====================================
# 1️-CHARGEMENT DES SOURCES MULTIPLES
# =====================================
def load_pdfs(pdf_folder):
    """Charge tous les fichiers PDF du dossier spécifié"""
    pdf_docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            loader = PyMuPDFLoader(file_path=path)
            pdf_docs.extend(loader.load())
            logging.info(f" PDF chargé : {file}")
    return pdf_docs


def load_corpus(corpus_path):
    """Charge un corpus JSON contenant les données collectées"""
    if not os.path.exists(corpus_path):
        logging.warning(f"Fichier corpus non trouvé : {corpus_path}")
        return []

    try:
        # Lecture du fichier corpus (JSON )
        if corpus_path.endswith(".json") or corpus_path.endswith(".csv"):
            df = pd.read_csv(corpus_path)
        else:
            raise ValueError("Format de fichier non supporté. Utilise .json")

        documents = []
        for i, row in df.iterrows():
            # On combine URL + Titre + Contenu comme texte principal
            content = f"{row.get('url', '')}\n{row.get('title', '')}\n{row.get('content', '')}\n{row.get('source', '')}"
            doc = Document(
                page_content=content,
                metadata={
                    "rating": row.get("Rating", None),
                    "date": row.get("Date", None),
                    "source": "corpus_web"
                },
                id=str(i)
            )
            documents.append(doc)
        logging.info(f"Corpus chargé ({len(documents)} documents)")
        return documents

    except Exception as e:
        logging.error(f" Erreur lors du chargement du corpus : {e}")
        return []


def load_text_source(source_path):
    """Charge un fichier texte brut (ex : sources ou articles non structurés)"""
    if not os.path.exists(source_path):
        logging.warning(f"Fichier source.txt introuvable")
        return []

    with open(source_path, "r", encoding="utf-8") as f:
        text = f.read()

    # On découpe le texte brut en petits morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    for d in docs:
        d.metadata = {"source": "source_txt"}

    logging.info(f"Fichier source.txt chargé ({len(docs)} morceaux)")
    return docs


# =====================================
# 2️ CRÉATION DE LA BASE VECTORIELLE
# =====================================
def build_or_load_vector_db(all_docs):
    """
    Crée ou charge la base vectorielle Chroma avec tous les documents disponibles.
    """
    db_exists = os.path.exists(VECTOR_DB_PATH)

    vector_store = Chroma(
        collection_name=VECTOR_COLLECTION,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    if not db_exists:
        logging.info(" Création d'une nouvelle base vectorielle...")
        ids = [str(i) for i in range(len(all_docs))]
        vector_store.add_documents(documents=all_docs, ids=ids)
        logging.info(f" {len(all_docs)} documents ajoutés à la base vectorielle.")
    else:
        logging.info("Base vectorielle existante chargée.")

    return vector_store


# =====================================
# 3️ RAG : RETRIEVER + LLM + PROMPTS
# =====================================
def create_retriever(vector_db, llm):
    """Crée un retriever multi-query pour des recherches contextuelles plus pertinentes"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=(
            "Tu es AgriBot Burkina, un assistant IA agricole spécialisé dans les pratiques locales. "
            "Génère cinq reformulations de la question suivante afin d'extraire les informations "
            "les plus pertinentes de la base de connaissances agricoles.\n\n"
            "Question d'origine : {question}"
        ),
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    return retriever


def create_chain(retriever, llm):
    """Crée la chaîne complète de génération RAG"""
    template = """Tu es AgriBot Burkina 🇧🇫, un assistant IA agricole fiable et bienveillant.
Tu aides les agriculteurs, étudiants et techniciens burkinabè à comprendre les bonnes pratiques agricoles
en te basant sur des données locales et des documents techniques.

Réponds uniquement avec les informations trouvées dans le contexte ci-dessous.
Si tu n'as pas assez d’information, réponds :
"Je n'ai pas trouvé cette information dans mes documents agricoles."

====================
Contexte :
{context}
====================

Question : {question}

Réponse claire et concise :
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# =====================================
# 4️ MAIN PIPELINE
# =====================================
def main():
    # 1. Charger toutes les sources de données
    pdf_docs = load_pdfs(PDF_PATH)
    corpus_docs = load_corpus(CORPUS_PATH)
    text_docs = load_text_source(SOURCE_PATH)

    # 2. Fusionner toutes les sources
    all_docs = pdf_docs + corpus_docs + text_docs
    logging.info(f"Total des documents fusionnés : {len(all_docs)}")

    # 3. Construire ou charger la base vectorielle
    vector_db = build_or_load_vector_db(all_docs)

    # 4. Initialiser le modèle de langage
    llm = ChatOllama(model=LLM_MODEL)

    # 5. Créer le retriever et la chaîne
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)

    # 6. Exemple de question
    question = "Comment améliorer la culture du maïs pendant la saison des pluies ?"
    response = chain.invoke(input=question)

    print("\n🤖 Réponse d’AgriBot Burkina :")
    print(response)


if __name__ == "__main__":
    main()
