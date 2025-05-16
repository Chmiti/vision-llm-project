import json

def clean_notebook(path_in, path_out):
    with open(path_in, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Supprimer les widgets et autres métadonnées inutiles
    metadata = data.get('metadata', {})
    metadata.pop('widgets', None)
    data['metadata'] = metadata

    for cell in data.get('cells', []):
        cell['metadata'].pop('widgets', None)
        cell['metadata'].pop('state', None)
        cell['metadata'].pop('execution', None)
        cell.pop('outputs', None)
        cell['execution_count'] = None

    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)

# Exemple d'utilisation
clean_notebook(
    "notebooks Google Colab/Vision_+_LLM_Project.ipynb",
    "notebooks Google Colab/Vision_+_LLM_Project_cleaned.ipynb"
)
