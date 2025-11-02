# AgriBot
AgriBot est un assistant intelligent contextuel 100 % open source, conçu au Burkina Faso pour soutenir les agriculteurs, étudiants et techniciens dans leurs activités. Son objectif est de promouvoir les bonnes pratiques agricoles en s’appuyant sur des données locales, des recherches scientifiques et des documents techniques nationaux.


---

## 🚀 Technologies principales

- **Embeddings** : `mxbai-embed-large` (modèle d'embedding performant).
- **LLM** : `mistral` (modèle open-source via Ollama).
- **Stack RAG** : LangChain + Chroma + Ollama.
- **API** : FastAPI.
- **Frontend** : HTML/JS minimal (fichier `index.html`) — peut être remplacé par React/Vue.


## ⚙️ Pré-requis

- Python 3.10+ (recommandé)
- `pip` ou `pipx`
- [Ollama](https://ollama.com/) installé localement
- Modèles Ollama (Mistral, embeddings) téléchargés localement (instructions ci-dessous)

---

## 📥 Installation des dépendances

Créer un environnement virtuel et installer les paquets :

```bash
python -m venv env
source env/bin/activate   # Linux / macOS
# env\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt


# Exemple : pull Mistral et le modèle d'embeddings (nom exact selon repo Ollama)
ollama pull mistral
ollama pull mxbai-embed-large




