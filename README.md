# AgriBot
AgriBot est un assistant intelligent contextuel 100 % open source, conçu au Burkina Faso pour soutenir les agriculteurs, étudiants et techniciens dans leurs activités. Son objectif est de promouvoir les bonnes pratiques agricoles en s’appuyant sur des données locales, des recherches scientifiques et des documents techniques nationaux.


---

## 🚀 Technologies principales

- **Embeddings** : `mxbai-embed-large` (modèle d'embedding performant).
- **LLM** : `llama3.2.1b` (modèle open-source via llama3.2:1b).
- **Stack RAG** : LangChain + Chroma + Ollama.
- **API** : FastAPI.
- **Frontend** : HTML/JS minimal (fichier `index.html`) — peut être remplacé par React/Vue.


## ⚙️ Pré-requis

- Python 3.10+ (recommandé)
- `pip` ou `pipx`
- [Ollama](https://ollama.com/) installé localement
- Modèles Ollama (llama3.2:1b, embeddings) téléchargés localement (instructions ci-dessous)

---

## 📥 Installation des dépendances

Créer un environnement virtuel et installer les paquets :

```bash
python -m venv env
source env/bin/activate   # Linux / macOS
# env\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt


# Exemple : pull Mistral et le modèle d'embeddings (nom exact selon repo Ollama)
ollama pull llama3.2:1b
ollama pull mxbai-embed-large

🚀 Démarrage rapide

Lancer l’API FastAPI :

uvicorn src.main:app --reload

🌾 Exemple d’utilisation

Une fois l’API démarrée :

Ouvre static/index.html

Saisis une question :
"Comment traiter les maladies du coton au Burkina Faso ?"

AgriBot recherche les documents pertinents et génère une réponse fiable et contextualisée.

🧪 Évaluation du modèle RAG

AgriBot inclut un script d’évaluation complet :

python src/rag-evaluate.py


🤝 Contribution

Les contributions sont bienvenues !

Fork le projet

Crée une branche (git checkout -b feature/nouvelle-fonctionnalite)

Commit (git commit -m "Ajout d’une nouvelle fonctionnalité")

Push (git push origin feature/nouvelle-fonctionnalite)

Ouvre une Pull Request

