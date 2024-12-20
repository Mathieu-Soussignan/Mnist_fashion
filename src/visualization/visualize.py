import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, labels, class_names, num_images=5, indices=None, normalize=False):
    """
    Affiche un certain nombre d'images avec leurs labels et noms de classes.

    Args:
        images : Tableau NumPy contenant les images
        labels : Tableau NumPy contenant les labels
        class_names : Liste des noms de classes
        num_images : Nombre d'images à afficher
        indices : Liste des indices des images à afficher (optionnel)
        normalize : Booléen, normalise les pixels à la plage [0, 1] si True
    """
    if indices is None:
        indices = np.arange(num_images)
    elif len(indices) != num_images:
        raise ValueError("Le nombre d'indices doit correspondre au nombre d'images à afficher.")

    if normalize:
        images = images / 255.0

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx], cmap=plt.cm.binary)
        plt.title(f"Label: {labels[idx]} ({class_names[labels[idx]]})")
        plt.axis('off')
    plt.show()

def plot_learning_curves(history, save_path=None):
    """
    Affiche les courbes d'apprentissage (accuracy et loss) à partir de l'historique d'entraînement Keras.

    Args:
        history : Objet History de Keras contenant l'historique d'entraînement
        save_path : Chemin pour sauvegarder les graphiques (optionnel)
    """
    if not all(key in history.history for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']):
        raise ValueError("L'historique ne contient pas les clés nécessaires ('accuracy', 'val_accuracy', 'loss', 'val_loss').")
    
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()