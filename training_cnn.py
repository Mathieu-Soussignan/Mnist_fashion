from tensorflow.keras.utils import to_categorical
from src.models.cnn import CNN  # Assurez-vous que le chemin vers votre classe CNN est correct
from src.data.load_data import load_fashion_mnist_data, preprocess_data, reshape_data

# Charger et prétraiter les données
(X_train, y_train, X_test, y_test) = load_fashion_mnist_data()
X_train, X_test = preprocess_data(X_train, X_test)
X_train, X_test = reshape_data(X_train, X_test)

# Encoder les étiquettes en one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Initialisation du modèle CNN
cnn = CNN()
cnn.creation_model()
cnn.compile_model()

# Entraîner et sauvegarder le modèle
save_path = "./src/models/cnn_model.h5"
cnn.train_model(X_train, y_train, epochs=10, validation_split=0.2, save_path=save_path)

print(f"Le modèle a été entraîné et sauvegardé à l'emplacement : {save_path}")