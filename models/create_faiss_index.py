import faiss
import numpy as np
import json

# Charger les embeddings CLIP
embeddings = np.load("clip_embeddings.npy").astype("float32")
print("Shape of embeddings:", embeddings.shape)

# Construire l'index FAISS (Inner Product = cosinus si normalisé)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Sauvegarder l'index
faiss.write_index(index, "faiss_index.index")

# Copier le fichier metadata (pour retrouver les descriptions ensuite)
with open("data/annotations.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ Index FAISS et metadata sauvegardés.")
