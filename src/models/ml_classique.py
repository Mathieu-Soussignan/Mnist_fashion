from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ML_Classique:
    def __init__(self, model_type="svm", **kwargs):
        """
        Initialise le modèle de Machine Learning classique.

        Args:
            model_type: str, type de modèle ("svm" ou "random_forest")
            **kwargs: paramètres pour le modèle choisi
        """
        if model_type == "svm":
            self.model = SVC(**kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Type de modèle non supporté")

    def train_model(self, X_train, y_train):
        """
        Entraîne le modèle sur les données fournies.

        Args:
            X_train: Les données d'entraînement
            y_train: Les étiquettes d'entraînement
        """
        self.model.fit(X_train, y_train)

    def predict_model(self, X):
        """
        Fait des prédictions sur les données fournies.

        Args:
            X: Les données sur lesquelles faire des prédictions

        Returns:
            Les prédictions du modèle
        """
        if not hasattr(self.model, "fit"):
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train_model() avant.")
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test, display_report=False):
        """
        Évalue le modèle sur les données de test.

        Args:
            X_test: Les données de test
            y_test: Les étiquettes de test
            display_report: bool, si True, affiche le rapport de classification

        Returns:
            accuracy: L'accuracy du modèle
            report: Le rapport de classification
        """
        y_pred = self.predict_model(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        if display_report:
            print("Accuracy:", accuracy)
            print("Rapport de classification:\n", report)
        
        return accuracy, report

    def save_model(self, filepath):
        """
        Sauvegarde le modèle entraîné dans un fichier.

        Args:
            filepath: Chemin où sauvegarder le modèle.
        """
        import joblib
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé depuis un fichier.

        Args:
            filepath: Chemin du fichier contenant le modèle sauvegardé.
        """
        import joblib
        self.model = joblib.load(filepath)

    def cross_validate_model(self, X, y, cv=5):
        """
        Effectue une validation croisée sur les données.

        Args:
            X: Les données
            y: Les étiquettes
            cv: Le nombre de folds pour la validation croisée

        Returns:
            scores: Les scores de validation croisée
        """
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores