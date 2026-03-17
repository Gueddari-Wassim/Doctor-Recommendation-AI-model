🩺 Doctor‑Recommendation AI Model

A specialty classification NLP model that predicts the most relevant medical specialty based on user‑provided symptom descriptions — built with PyTorch and deployed via FastAPI.

This model serves as the core AI powerhouse for medical orientation systems, helping connect users to the most appropriate medical domain based on textual symptoms.

📌 Repository Purpose

This repository holds:

A fine‑tuned text classification model (specialty_model_v2)

A REST API to serve the model for inference

Lightweight code for symptom → medical specialty recommendation

It does not include a frontend — it’s meant to be used as an independent microservice for AI inference.

🧠 Model Overview

Task: Text‑based medical specialty recommendation

Framework: PyTorch + Transformers

Deployment: FastAPI API

Input: User symptom description (plain text)

Output: Top‑k predicted medical specialty labels with confidence scores

The model processes natural language text and outputs the most probable medical specialty (e.g., Cardiology, Dermatology, Pulmonology, etc.), which can drive doctor recommendation systems and triage workflows.

🚀 Quick Start
📥 Clone
git clone https://github.com/Gueddari-Wassim/Doctor‑Recommendation‑AI‑model.git
cd Doctor‑Recommendation‑AI‑model
🛠️ Installation

Create a Python virtual environment:

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

⚠️ Ensure you have PyTorch installed compatible with your system’s CUDA/CPU configuration.

🚀 Running the API

Start FastAPI:

uvicorn main:app --reload

By default, the API runs on http://localhost:8000.

Test it with a POST request:

curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"text": "I have chest pain and shortness of breath", "k": 3}'

It should return something like:

{
  "input": "I have chest pain and shortness of breath",
  "status": "ok",
  "top": [
    {"label": "Cardiology", "confidence": 0.87},
    {"label": "Pulmonology", "confidence": 0.09},
    {"label": "Internal_Medicine", "confidence": 0.03}
  ],
  "threshold": 0.4
}
🧩 Core Logic (Inference)

The API works by:

Tokenizing input text

Running the model to obtain logits

Applying softmax to get probabilities

Returning the top‑k predictions with confidence scores

Using a confidence threshold (CONF_THRESH = 0.4) to handle uncertainty

# Example snippet from main.py
logits = model(**inputs).logits
probs = softmax(logits, dim=-1).squeeze(0)

This design ensures fast, reliable inference and supports integration into larger backend services, mobile apps, or dashboards.

🧠 Integrations

This model + API can be used in:

Telemedicine systems

Symptom checking apps

Doctor recommendation services

Triage workflows

Chatbot assistants

It can be deployed in cloud environments, Docker containers, or serverless functions.

📌 Disclaimer

This tool does not replace professional medical advice. Predictions are based on learned patterns from training data and should be used to support, not substitute, clinical judgment.

🛠️ Future Improvements

Suggestions:

Add multilingual support

Upgrade to transformer‑based encoders (e.g., BERT)

Add confidence calibration

Integrate additional medical knowledge sources

Publish evaluation on a public test set

⭐ Contributing

Contributions are welcome!

Fork

Create a branch

Submit a pull request

📬 Contact

GitHub: https://github.com/Gueddari‑Wassim

Email: wassimgueddari13@gmail.com
