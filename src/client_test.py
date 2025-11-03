
import requests

API_URL = "http://127.0.0.1:8000/generate"
API_KEY = "agri_bot"

def poser_question(question):
    print(f"🤔 Question : {question}")
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {"prompt": question}

    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        res = response.json()
        print("💬 Réponse:", res["response"])
    else:
        print("❌ Erreur:", response.text)

if __name__ == "__main__":
    poser_question("Comment cultiver le mil au Burkina Faso ?")

