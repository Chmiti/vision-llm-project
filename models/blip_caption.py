import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main(image_folder="data/images"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)
            caption = generate_caption(image_path, processor, model, device)
            print(f"{filename} → {caption}")
            results.append((filename, caption))

    # (Optionnel) Enregistrer dans un fichier
    with open("data/captions.csv", "w", encoding="utf-8") as f:
        f.write("filename,caption\n")
        for fname, cap in results:
            f.write(f"{fname},{cap}\n")

if __name__ == "__main__":
    main()
