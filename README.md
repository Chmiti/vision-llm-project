# 🧠 Vision-Language Intelligent QA System

Un projet de génération de légendes d'images, question-réponse visuelle et raisonnement assisté par LLM, combinant les modèles BLIP, CLIP, FAISS et une interface Gradio.

---

## 📸 Fonctionnalités principales

- 🔍 Génération de légendes à partir d’images (BLIP)
- ❓ Question-Réponse visuelle avec prompt engineering
- 🧠 Raisonnement intelligent à l’aide d’un LLM (OpenAI GPT)
- 📚 Système RAG : Recherche augmentée via CLIP + FAISS
- 🖥️ Interface interactive via Gradio

---

## 🛠️ Stack technique

| Composant        | Rôle                                          |
|------------------|-----------------------------------------------|
| `BLIP`           | Génération automatique de légendes d’images   |
| `CLIP`           | Encodage visuel et recherche d’images similaires |
| `FAISS`          | Indexation rapide pour recherche de similarité |
| `OpenAI GPT`     | Raisonnement et génération de réponses         |
| `Gradio`         | Interface web simple et rapide                 |
| `Python`         | Langage principal (avec PyTorch, Transformers) |

---

## 🧪 Démo rapide

1. 📤 Upload une image
2. ✍️ Pose une question (ex: *Quel est ce panneau ?*)
3. 🤖 Le système :
   - Génère une légende avec BLIP
   - Cherche des images similaires avec CLIP + FAISS
   - Envoie le tout à GPT pour obtenir une réponse contextualisée
4. 💬 Réponse affichée via Gradio

---

## 🔧 Lancer le projet

```bash
# Crée ton environnement
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

# Installe les dépendances
pip install -r requirements.txt

# Lance l'app Gradio
python app.py
📁 Organisation du projet
bash
Copy
Edit
VISION-LLM-PROJECT/
│
├── app/                  # Fichiers principaux d’inférence
├── models/               # Scripts pour BLIP, CLIP, FAISS
├── pipeline/             # Étapes de pipeline
├── data/                 # Images, index FAISS, métadonnées
├── .gradio/ .env .gitignore requirements.txt README.md
```
⚠️ Limitations actuelles et ouverture:

🎯 Précision variable des captions générées (manque d’adaptation au domaine)

📉 Qualité des réponses dépend fortement de la base CLIP + FAISS

🧊 Temps de réponse élevé sans optimisation hardware (GPU)

🧠 LLM non fine-tuné sur les requêtes spécifiques du domaine

🔜 Ces limites m’ont inspiré un projet plus ambitieux, où je vais :

Fine-tuner BLIP et CLIP sur un dataset spécialisé (ex. panneaux de signalisation, objets médicaux…)

Créer une base multimodale enrichie (image + texte + tags)

Intégrer un LLM open-source finement ajusté pour du Q&A intelligent

Optimiser le système avec quantization / onnx / GPU / Docker

💡 Ce projet servira donc de fondation pour une version plus robuste, plus rapide et plus précise d’un système de question-réponse visuelle intelligent.

👤 Auteur
Taha Chmiti
Élève-ingénieur à l’ENSEIRB-Matmeca
Spécialisé en Traitement d’image et Intelligence Artificielle
📫 taha.chmiti@enseirb-matmeca.fr

⭐ Pourquoi ce projet ?
Ce projet montre ma capacité à :

Intégrer plusieurs modèles IA dans un pipeline cohérent

Concevoir un système multi-modèle basé sur l'image et le langage

Construire une interface interactive et exploitable en local

Prendre du recul sur les limites et identifier des perspectives concrètes