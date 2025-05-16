from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import csv

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# ✅ Chargement sans bitsandbytes, pour CPU uniquement
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # pas de float16 si pas de GPU
    device_map="auto",
    low_cpu_mem_usage=True  # important pour éviter de saturer la RAM
)

# Générateur de texte
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def ask_mistral(prompt):
    output = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.5)[0]["generated_text"]
    return output[len(prompt):].strip()

def main(captions_file="data/captions.csv", output_file="data/reasoning.csv"):
    print("✅ main() started")

    with open(captions_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(output_file, "w", newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["filename", "caption", "llm_response"])

        for row in rows:
            prompt = f"""### Instruction:
You are a helpful assistant analyzing traffic images.

### Caption:
{row['caption']}

### Question:
What do you infer about this image?

### Answer:"""
            print(f"\n🛠️ Prompt : {prompt[:100]}...")
            answer = ask_mistral(prompt)
            print(f"🖼️ {row['filename']}\n📜 {row['caption']}\n🤖 {answer}")
            writer.writerow([row['filename'], row['caption'], answer])

if __name__ == "__main__":
    main()
