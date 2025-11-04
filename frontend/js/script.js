const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const chatWindow = document.getElementById('chat-window');
const historyList = document.getElementById('history-list');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const userMessage = input.value;
  appendMessage(userMessage, 'user');
  addToHistory(userMessage);
  input.value = '';

  const botMessage = await getBotResponse(userMessage);
  appendMessage(botMessage, 'bot');
});

function appendMessage(message, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message');
  msgDiv.classList.add(sender);
  msgDiv.textContent = message;
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addToHistory(message) {
  const item = document.createElement('li');
  item.textContent = message;
  historyList.appendChild(item);
}

function quickAsk(message) {
  input.value = message;
  form.dispatchEvent(new Event('submit'));
}

async function getBotResponse(message) {
  const API_URL = 'http://127.0.0.1:8000/generate'; // Endpoint FastAPI

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt: message }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Erreur API:", errorText);
      return "⚠️ Erreur côté serveur.";
    }

    const data = await response.json();
    return data.response || "🤖 Aucune réponse générée.";
  } catch (error) {
    console.error("Erreur de connexion:", error);
    return "🚫 Impossible de contacter le serveur.";
  }
}