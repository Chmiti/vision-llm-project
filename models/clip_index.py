import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Chargement du modèle CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Charger les annotations
with open("data/annotations.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

# Dossier des images
image_dir = "data"

# Initialisation des résultats
embeddings = []
valid_metadata = []

# Boucle sur les images
for ann in annotations:
    path = os.path.join(image_dir, ann["filename"])
    if not os.path.exists(path):
        print(f"❌ Image not found: {path}")
        continue

    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy()[0])
        valid_metadata.append(ann)

# Sauvegarde
embeddings_np = np.array(embeddings)
np.save("clip_embeddings.npy", embeddings_np)

with open("dataannotations.json", "w", encoding="utf-8") as f:
    json.dump(valid_metadata, f, indent=2)

print("✅ Embeddings saved:", embeddings_np.shape)
