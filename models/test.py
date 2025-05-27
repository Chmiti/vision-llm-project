from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()  # Il récupère automatiquement la clé depuis OPENAI_API_KEY


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, what is 2+2?"}]
)

print(response.choices[0].message.content)
