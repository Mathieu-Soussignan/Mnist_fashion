import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def flatten_data(X):
    """
    Aplatit les données d'images en vecteurs pour les modèles de ML classique.

    Args:
        X : tableau NumPy contenant les données d'images (ex: (60000, 28, 28))

    Returns:
        X_flat : tableau NumPy contenant les données aplaties (ex: (60000, 784))
    """
    if len(X.shape) != 3:
        raise ValueError("Les données doivent avoir trois dimensions : (échantillons, hauteur, largeur).")
    return X.reshape(X.shape[0], -1)

def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle sur les données de test et affiche les métriques.

    Args:
        model : Le modèle entraîné
        X_test : Les données de test
        y_test : Les étiquettes de test
    """
    if not hasattr(model, "predict"):
        raise ValueError("L'objet fourni n'est pas un modèle valide.")
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Affiche la matrice de confusion.

    Args:
        y_true : Les vraies étiquettes
        y_pred : Les étiquettes prédites
        class_names : Les noms des classes
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annoter les cases avec les valeurs
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie valeur')
    plt.tight_layout()
    plt.show()