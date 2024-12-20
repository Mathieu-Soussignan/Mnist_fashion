from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class ReseauDense:
    def __init__(self, num_classes=10, input_shape=(784,)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None 

    def creation_model(self):
        """
        Crée l'architecture du modèle de réseau de neurones dense.
        """
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')  # Couche de sortie
        ])

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        """
        Compile le modèle avec la fonction de perte, l'optimiseur et les métriques spécifiés.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de compiler.")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def encode_labels(self, y):
        """
        Encode les labels sous forme one-hot.

        Args:
            y: Les labels à encoder

        Returns:
            Les labels encodés
        """
        return to_categorical(y, self.num_classes)

    def train_model(self, X_train, y_train, epochs=10, validation_split=0.2):
        """
        Entraîne le modèle sur les données fournies.

        Args:
            X_train: Les données d'entraînement
            y_train: Les étiquettes d'entraînement encodées sous forme one-hot
            epochs: Le nombre d'epochs pour l'entraînement
            validation_split: La proportion des données d'entraînement à utiliser pour la validation
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de l'entraîner.")
        y_train = self.encode_labels(y_train)  # Encodage one-hot
        self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)

    def predict_model(self, X):
        """
        Fait des prédictions sur les données fournies.

        Args:
            X: Les données sur lesquelles faire des prédictions

        Returns:
            Les prédictions du modèle (probabilités pour chaque classe)
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de prédire.")
        return self.model.predict(X)

    def save_model(self, filepath):
        """
        Sauvegarde le modèle entraîné dans un fichier.

        Args:
            filepath: Chemin où sauvegarder le modèle.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Rien à sauvegarder.")
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé depuis un fichier.

        Args:
            filepath: Chemin du fichier contenant le modèle sauvegardé.
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)

    def plot_history(self):
        """
        Affiche les courbes d'apprentissage (accuracy et loss) en fonction des epochs.
        """
        if self.history is None:
            raise ValueError("Aucun historique trouvé. Entraînez le modèle avant d'afficher les courbes.")
        
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Accuracy en fonction des epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Loss en fonction des epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='upper left')
        plt.show()