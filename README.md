# 🧠 Vision + LLM Project — Raisonner sur des images avec l'IA générative

Ce projet explore l'utilisation des **Large Language Models (LLMs)** et des outils de **vision par ordinateur** pour analyser, décrire et interagir avec des images. Il combine des modèles de vision (BLIP-2, CLIP, LLaVA) avec des LLMs (GPT, Mistral, etc.) pour développer un raisonnement multimodal.

---

## 🎯 Objectifs

- Générer automatiquement des **légendes d'images** (vision → texte)
- Poser des **questions sur une image** (prompt engineering)
- Répondre intelligemment avec un **LLM**
- (Bonus) Ajouter un **système RAG** : recherche augmentée avec `CLIP` + `FAISS`
- Créer une interface interactive via **Gradio**

---

## 🧱 Structure du projet

vision-llm-project/
├── data/ # Images d'entrée, descriptions, bases

├── models/ # Scripts BLIP, CLIP, LLaVA

├── pipeline/ # Traitement complet : image → réponse

├── app/ # Interface utilisateur (Gradio)

├── requirements.txt # Dépendances Python

├── .gitignore # Fichiers à ignorer par Git

└── README.md # Description du projet


---

## 🔧 Technologies utilisées

- 🖼️ `BLIP-2`, `CLIP`, `LLaVA`
- 🤖 `transformers`, `OpenAI`, `Mistral`
- 🧠 `PyTorch`, `sentencepiece`, `accelerate`
- 🔍 `FAISS`, `Chroma` (RAG)
- 💻 `Gradio` (interface)

---

## 🚧 Avancement

- ✅ Initialisation du dépôt et de la structure
- ✅ Environnement Python + dépendances
- ⏳ Génération de légende avec BLIP-2
- ⏳ Intégration d’un LLM pour raisonnement
- ⏳ Création de l’interface utilisateur
- ⏳ (optionnel) Ajout d’un module RAG
- ⏳ (optionnel) Fine-tuning d’un mini LLM

---

## 📸 Exemple de pipeline

1. Upload d’une image
2. Génération de la description (`BLIP-2`)
3. Construction d’un prompt : *"Voici une scène : [description]. Que peut-on en conclure ?"*
4. Réponse générée par un LLM
5. (Bonus) Recherche d’images similaires et injection dans le prompt (RAG)

---

## 📄 Licence

Projet personnel open-source – libre d’utilisation à but pédagogique.

---

## 👤 Auteur

Taha Chmiti – Élève-ingénieur en électronique, spécialisé en IA et traitement d’image  
🔗 [LinkedIn](https://www.linkedin.com/in/taha-chmiti/)  
📫 taha.chmiti@etu.enseirb-matmeca.fr

