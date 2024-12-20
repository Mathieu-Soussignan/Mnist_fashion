import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle
@st.cache_resource
def load_keras_model(model_path: str):
    """
    Charge un modèle Keras depuis un fichier.

    Args:
        model_path : Chemin vers le modèle Keras.

    Returns:
        Le modèle Keras chargé.
    """
    try:
        logger.info(f"Chargement du modèle depuis {model_path}")
        return load_model(model_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        st.error("Impossible de charger le modèle.")
        st.stop()

model = load_keras_model("./src/models/cnn_model.h5")

# Charger le dataset Fashion-MNIST
@st.cache_resource
def load_fashion_mnist_data():
    """
    Charge le dataset Fashion-MNIST et retourne les données de test.

    Returns:
        X_test, y_test : Images et étiquettes de test.
    """
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test / 255.0  # Normaliser les données
    X_test = X_test.reshape(-1, 28, 28, 1)  # Ajouter une dimension pour le canal
    return X_test, y_test

X_test, y_test = load_fashion_mnist_data()

# Interface Streamlit
st.title("Prédictions avec CNN - Fashion MNIST")

# Affichage d'une image aléatoire du dataset
st.subheader("Image tirée du dataset Fashion MNIST")
random_index = st.slider("Choisissez une image (index dans le dataset)", 0, len(X_test) - 1, 0)

# Afficher l'image sélectionnée
image = X_test[random_index].reshape(28, 28)
true_label = y_test[random_index]
st.image(image, caption=f"Label réel : {true_label}", width=150, channels="gray")

# Bouton pour effectuer la prédiction
if st.button("Prédire la classe"):
    # Prédire la classe
    preprocessed_image = X_test[random_index].reshape(1, 28, 28, 1)  # Extraire l'image au bon format
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Afficher les résultats
    st.write(f"Classe prédite : {predicted_class}")
    st.write(f"Probabilités des classes : {np.round(prediction[0], 2)}")

    # Afficher un graphique des probabilités
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xlabel("Classes")
    ax.set_ylabel("Probabilité")
    ax.set_title("Distribution des probabilités")
    st.pyplot(fig)