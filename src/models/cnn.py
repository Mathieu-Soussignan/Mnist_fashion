import matplotlib.pyplot as plt
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

class CNN:
    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def creation_model(self):
        """
        Crée l'architecture du modèle CNN.
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

    def compile_model(self, loss='categorical_crossentropy', optimizer=None, metrics=['accuracy']):
        """
        Compile le modèle avec la fonction de perte, l'optimiseur et les métriques spécifiés.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de compiler.")
        if optimizer is None:
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam()  # Optimiseur par défaut
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, X_train, y_train, epochs=10, validation_split=0.2, save_path=None):
        """
        Entraîne le modèle sur les données fournies avec suivi MLflow.

        Args:
            X_train: Les données d'entraînement
            y_train: Les étiquettes d'entraînement encodées sous forme one-hot
            epochs: Le nombre d'epochs pour l'entraînement
            validation_split: La proportion des données d'entraînement à utiliser pour la validation
            save_path: Chemin pour sauvegarder le modèle après entraînement (optionnel)
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de l'entraîner.")

        # Vérifie s'il existe déjà une run active
        if mlflow.active_run():
            mlflow.end_run()

        # Suivi MLflow
        with mlflow.start_run():
            # Log des hyperparamètres
            mlflow.log_param("model_type", "CNN")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("validation_split", validation_split)

            # Entraînement du modèle
            self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=32)

            # Log des métriques
            training_accuracy = self.history.history['accuracy'][-1]
            validation_accuracy = self.history.history['val_accuracy'][-1]
            training_loss = self.history.history['loss'][-1]
            validation_loss = self.history.history['val_loss'][-1]

            mlflow.log_metric("training_accuracy", training_accuracy)
            mlflow.log_metric("validation_accuracy", validation_accuracy)
            mlflow.log_metric("training_loss", training_loss)
            mlflow.log_metric("validation_loss", validation_loss)

            # Sauvegarde du modèle
            if save_path:
                self.save_model(save_path)
                mlflow.log_artifact(save_path)  # Enregistre le modèle dans MLflow
                print(f"Modèle sauvegardé à l'emplacement : {save_path}")

            # Sauvegarde des artefacts (courbes d'apprentissage)
            plot_path = "training_history.png"
            self.plot_history(save_as=plot_path)
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)  # Supprime le fichier local après sauvegarde

    def train_model_with_augmentation(self, X_train, y_train, epochs=10, validation_split=0.2, save_path=None):
        """
        Entraîne le modèle avec augmentation des données.

        Args:
            X_train: Données d'entraînement
            y_train: Étiquettes d'entraînement
            epochs: Nombre d'epochs
            validation_split: Proportion des données de validation
            save_path: Chemin pour sauvegarder le modèle
        """
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False
        )
        datagen.fit(X_train)

        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez creation_model() avant de l'entraîner.")

        # Entraîner avec augmentation
        steps_per_epoch = X_train.shape[0] // 32  # Batch size par défaut
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=steps_per_epoch,
            validation_split=validation_split,
            epochs=epochs
        )

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
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Rien à sauvegarder.")
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé depuis un fichier.
        """
        from tensorflow.keras.models import load_model
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
        self.model = load_model(filepath)

    def evaluate_model(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test et affiche les résultats.

        Args:
            X_test: Données de test
            y_test: Labels de test
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé ou chargé. Appelez creation_model() ou load_model().")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results

    def plot_history(self, save_as=None):
        """
        Affiche les courbes d'apprentissage (accuracy et loss) en fonction des epochs.

        Args:
            save_as: Nom du fichier pour sauvegarder les courbes (optionnel)
        """
        if self.history is None:
            raise ValueError("Aucun historique trouvé. Entraînez le modèle avant d'afficher les courbes.")

        # Plot Accuracy
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='Entraînement')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy en fonction des epochs')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        if save_as:
            plt.savefig(save_as)
        plt.show()
        plt.close()

        # Plot Loss
        plt.figure()
        plt.plot(self.history.history['loss'], label='Entraînement')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Loss en fonction des epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        if save_as:
            plt.savefig(save_as)
        plt.show()
        plt.close()