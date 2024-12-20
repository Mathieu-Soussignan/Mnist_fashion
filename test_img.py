from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Charger le dataset Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normaliser les données
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaper les données (ajout du canal pour CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Encoder les labels en one-hot
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

# Charger le modèle entraîné
model = load_model("./src/models/cnn_model.h5")

# Tester sur quelques images d'entraînement
def test_predictions(model, X_data, y_labels, num_samples=5):
    """
    Teste les prédictions du modèle sur un échantillon des données fournies.

    Args:
        model: Le modèle chargé.
        X_data: Les données d'entrée (images).
        y_labels: Les labels correspondants.
        num_samples: Nombre d'images à tester.

    Returns:
        Affiche les images et leurs prédictions.
    """
    indices = np.random.choice(len(X_data), num_samples, replace=False)
    for idx in indices:
        img = X_data[idx]
        label = y_labels[idx]
        
        # Prédiction du modèle
        prediction = model.predict(img.reshape(1, 28, 28, 1))
        predicted_class = np.argmax(prediction)
        
        # Afficher l'image et la prédiction
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Vrai label: {label}, Prédiction: {predicted_class}")
        plt.axis("off")
        plt.show()

# Tester avec des données d'entraînement
print("Test avec les données d'entraînement:")
test_predictions(model, X_train, y_train, num_samples=5)

# Tester avec des données de test
print("Test avec les données de test:")
test_predictions(model, X_test, y_test, num_samples=5)