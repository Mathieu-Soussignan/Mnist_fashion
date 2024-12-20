import requests
import numpy as np

# Générer une image aléatoire 28x28
random_image = np.random.randint(0, 256, (28, 28)).tolist()

# URL de l'API
url = "http://127.0.0.1:8000/predict"

# Requête POST
response = requests.post(url, json={"image": random_image})

# Résultat
print(response.status_code)
print(response.json())