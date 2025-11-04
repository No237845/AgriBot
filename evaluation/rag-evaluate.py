# =========================================================
# 🌾 AgriBot Burkina - Évaluation du système RAG
# =========================================================
# Ce script évalue ton modèle RAG selon trois critères :
#   1️⃣ Pertinence des documents récupérés
#   2️⃣ Fidélité de la réponse (absence d’hallucination)
#   3️⃣ Exactitude de la réponse (par rapport à la vérité attendue)
#
# ⚙️ 100 % open-source : utilise des modèles SentenceTransformer,
# BERTScore et ROUGE sans dépendance OpenAI.
#
# Auteur : Kabore Innocent
# =========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from evaluate import load

# =========================================================
# 1️⃣ CHARGEMENT DES DONNÉES
# =========================================================
# Le fichier d’entrée doit contenir au minimum :
# - "question" : la question posée à ton RAG
# - "retrieved_doc" : le passage extrait par le retriever
# - "generated_answer" : la réponse produite par ton modèle
# - "expected_answer" : la réponse de référence (si disponible)

EVAL_DATA_PATH = "./evaluation/rag_eval_dataset.csv"

if not os.path.exists(EVAL_DATA_PATH):
    raise FileNotFoundError(f"❌ Fichier d’évaluation introuvable : {EVAL_DATA_PATH}")

df = pd.read_csv(EVAL_DATA_PATH)
print(f"✅ Données chargées : {len(df)} exemples\n")

# =========================================================
# 2️⃣ ÉVALUATION DE LA PERTINENCE (retrieval relevance)
# =========================================================
# On mesure la similarité entre la question et le document
# récupéré à l’aide d’un modèle de similarité sémantique.
# Valeurs proches de 1 = document très pertinent.

print("🔹 Évaluation de la pertinence des documents...")

model_sim = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def relevance_score(question, retrieved_doc):
    emb = model_sim.encode([question, retrieved_doc])
    return float(util.cos_sim(emb[0], emb[1]))

df["relevance"] = df.apply(
    lambda r: relevance_score(r["question"], r["retrieved_doc"]), axis=1
)

# =========================================================
# 3️⃣ ÉVALUATION DE LA FIDÉLITÉ (faithfulness)
# =========================================================
# On utilise BERTScore pour mesurer la cohérence entre
# la réponse générée et le document source. Cela permet de
# détecter les "hallucinations".
# Valeur proche de 1 = réponse fidèle au contexte.

print("🔹 Évaluation de la fidélité des réponses...")

bertscore = load("bertscore")
faithfulness = bertscore.compute(
    predictions=df["generated_answer"].tolist(),
    references=df["retrieved_doc"].tolist(),
    lang="fr"
)
df["faithfulness"] = faithfulness["f1"]

# =========================================================
# 4️⃣ ÉVALUATION DE L’EXACTITUDE (answer correctness)
# =========================================================
# On compare la réponse générée à la "bonne" réponse
# attendue avec la métrique ROUGE-L.
# Plus le score est haut, plus la réponse est correcte.

print("🔹 Évaluation de l’exactitude des réponses...")

rouge = load("rouge")
rouge_results = rouge.compute(
    predictions=df["generated_answer"].tolist(),
    references=df["expected_answer"].tolist()
)
df["rougeL"] = rouge_results["rougeL"]

# =========================================================
# 5️⃣ COMBINAISON DES MÉTRIQUES
# =========================================================
# Pondération : pertinence (40%) + fidélité (30%) + exactitude (30%)
# Ce score global permet de classer la performance du système RAG.

df["global_score"] = (
    0.4 * df["relevance"] +
    0.3 * df["faithfulness"] +
    0.3 * df["rougeL"]
)

# =========================================================
# 6️⃣ ANALYSE & VISUALISATION
# =========================================================
print("\n📊 Résumé des scores moyens :")
print(f"- Pertinence moyenne   : {df['relevance'].mean():.3f}")
print(f"- Fidélité moyenne     : {df['faithfulness'].mean():.3f}")
print(f"- Exactitude moyenne   : {df['rougeL'].mean():.3f}")
print(f"- Score global moyen   : {df['global_score'].mean():.3f}")

# 🔹 Affichage graphique
plt.figure(figsize=(10, 5))
plt.bar(df["question"], df["global_score"], color="#4CAF50")
plt.xticks(rotation=45, ha="right")
plt.title("🌾 Évaluation globale du système RAG AgriBot Burkina")
plt.ylabel("Score (0 à 1)")
plt.tight_layout()
plt.show()

# 🔹 Sauvegarde du rapport
OUTPUT_PATH = "./data/eval_results.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n📁 Résultats enregistrés dans : {OUTPUT_PATH}")