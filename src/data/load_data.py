from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist_data():
    """
    Charge le jeu de données Fashion-MNIST et le divise en ensembles d'entraînement et de test.

    Retourne:
        X_train, y_train, X_test, y_test : tableaux NumPy contenant les données et les labels
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    """
    Prétraite les données en les normalisant.

    Args:
        X_train, X_test : tableaux NumPy contenant les données d'entraînement et de test

    Retourne:
        X_train, X_test : tableaux NumPy contenant les données prétraitées
    """
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

def reshape_data(X_train, X_test):
    """
    Remodeler les données pour les modèles CNN (ajout d'un canal de couleur).

    Args:
        X_train, X_test : tableaux NumPy contenant les données d'entraînement et de test

    Retourne:
        X_train, X_test : tableaux NumPy contenant les données remodelées
    """
    X_train = X_train.reshape(-1, 28, 28, 1)  
    X_test = X_test.reshape(-1, 28, 28, 1)
    return X_train, X_test 

def flatten_data(X_train, X_test):
    """
    Aplatir les données pour les modèles de Machine Learning classique (28x28 -> 784).

    Args:
        X_train, X_test : tableaux NumPy contenant les données d'entraînement et de test

    Retourne:
        X_train, X_test : tableaux NumPy contenant les données aplaties
    """
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, X_test

def load_and_prepare_data(model_type="cnn"):
    """
    Charge et prétraite les données Fashion-MNIST en fonction du type de modèle.

    Args:
        model_type : str, "cnn" ou "ml_classique" pour déterminer le prétraitement

    Retourne:
        X_train, y_train, X_test, y_test : tableaux NumPy
    """
    X_train, y_train, X_test, y_test = load_fashion_mnist_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    
    if model_type == "cnn":
        X_train, X_test = reshape_data(X_train, X_test)
    elif model_type == "ml_classique":
        X_train, X_test = flatten_data(X_train, X_test)
    else:
        raise ValueError("Type de modèle non supporté : 'cnn' ou 'ml_classique' attendu")
    
    return X_train, y_train, X_test, y_test

# Exemple d'utilisation
if __name__ == "__main__":
    model_type = "cnn"
    X_train, y_train, X_test, y_test = load_and_prepare_data(model_type=model_type)
    print(f"Données prêtes pour un modèle {model_type}.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")