import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
from openai import OpenAI

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📦 Charger la clé API OpenAI depuis .env
load_dotenv()
client = OpenAI()  # utilisera automatiquement OPENAI_API_KEY

# 🔧 Initialisation des modèles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 BLIP (image → caption)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 🧠 CLIP (image → embedding)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 📂 Charger l’index FAISS et les descriptions
faiss_index = faiss.read_index("faiss_index.index")
with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    clip_metadata = json.load(f)

# 🚀 Pipeline principal
def pipeline(image_pil):
    try:
        # 1. 🖼 Génération de la caption (BLIP)
        image_pil = image_pil.convert("RGB")
        inputs = blip_processor(images=image_pil, return_tensors="pt").to(device)
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

        # 2. 🔎 Embedding avec CLIP
        clip_inputs = clip_processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            image_feat = clip_model.get_image_features(**clip_inputs)
            image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
        query = image_feat.cpu().numpy()

        # 3. 🔁 Recherche dans FAISS
        D, I = faiss_index.search(query, k=5)
        similar_descriptions = [clip_metadata[i]["description"] for i in I[0]]

        # 4. 🤖 Prompt GPT
        prompt = f"""Voici un panneau routier détecté.

Caption automatique : "{caption}"

Panneaux similaires dans la base :
{chr(10).join(f"- {desc}" for desc in similar_descriptions)}

👉 Que signifie probablement ce panneau ? Réponds de façon claire et concise."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()

        # ✅ Résultat formaté
        return caption, "\n".join(["- " + d for d in similar_descriptions]), answer

    except Exception as e:
        import traceback
        return "Erreur", "Erreur", f"❌ Exception :\n{traceback.format_exc()}"

# 🌐 Interface Gradio
iface = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil", label="Upload a road sign image"),
    outputs=[
        gr.Textbox(label="📷 Caption"),
        gr.Textbox(label="🔎 Similar Signs"),
        gr.Textbox(label="🧠 GPT Reasoning")
    ],
    title="🛣️ Road Sign Analyzer with RAG + GPT",
    description="Upload a road sign image. The system uses BLIP (caption), CLIP+FAISS (retrieval), and GPT for reasoning."
)

if __name__ == "__main__":
    iface.launch()
