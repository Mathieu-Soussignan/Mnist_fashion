# Projet MNIST Fashion - README

## Structure du projet

Voici la structure détaillée de votre projet avec une description de chaque fichier et répertoire :

```
PROJET_MNIST_FASHION/
├── __pycache__/                # Fichiers de cache Python générés automatiquement
├── .venv/                      # Environnement virtuel Python
├── generated_images/           # Dossier contenant les images générées pour les tests
├── mlruns/                     # Répertoire de suivi MLflow
├── notebooks/                  # Dossier contenant les notebooks pour l'exploration et les modèles
│   ├── 01_exploration.ipynb            # Analyse exploratoire des données MNIST
│   ├── 02_model_ml_classique.ipynb     # Implémentation des modèles de machine learning classique
│   ├── 03_model_reseau_dense.ipynb     # Implémentation du modèle de réseau dense (MLP)
│   ├── 04_model_cnn.ipynb              # Implémentation et entraînement du modèle CNN
│   └── 05_mlflow_tracking.ipynb        # Suivi des expériences avec MLflow
├── src/                       # Répertoire principal pour le code source
│   ├── data/                  # Scripts pour la gestion des données
│   │   └── load_data.py       # Chargement et prétraitement des données MNIST Fashion
│   ├── models/                # Scripts pour les modèles
│   │   ├── cnn.py             # Définition et entraînement du modèle CNN
│   │   ├── ml_classique.py    # Implémentation des modèles de machine learning classique (SVM, Random Forest)
│   │   └── reseau_dense.py    # Implémentation du réseau dense (MLP)
│   └── visualization/         # Scripts pour la visualisation des données et des résultats
│       └── visualize.py       # Visualisation des images et des courbes d'entraînement
├── app.py                     # Interface Streamlit pour tester le modèle avec des images personnalisées ou dessinées
├── create_img.py              # Script pour générer des images synthétiques pour les tests
├── Readme.md                  # Documentation du projet
├── requirements.txt           # Liste des dépendances Python nécessaires au projet
├── test_api.py                # Script pour tester l'API FastAPI
├── test_img.py                # Script pour visualiser les images générées
└── training_cnn.py            # Script pour entraîner le modèle CNN et sauvegarder le modèle final
```

## Explications détaillées

### Dossiers principaux

- **`generated_images/`** : Contient des images synthétiques générées par `create_img.py` pour tester les prédictions du modèle.
- **`mlruns/`** : Répertoire utilisé par MLflow pour suivre les expériences et les résultats d'entraînement.
- **`notebooks/`** : Contient des notebooks Jupyter pour chaque étape du projet, notamment l'analyse exploratoire, l'implémentation des modèles, et le suivi avec MLflow.
- **`src/`** : Dossier contenant tous les scripts Python nécessaires à l'exécution du projet.

### Fichiers principaux

- **`app.py`** :
  - Fournit une interface utilisateur interactive avec Streamlit.
  - Permet de tester le modèle en chargeant une image ou en dessinant directement sur un canvas.

- **`create_img.py`** :
  - Génère des images synthétiques pour tester les prédictions du modèle.
  - Utile pour valider le comportement du modèle avec des données générées artificiellement.

- **`training_cnn.py`** :
  - Entraîne le modèle CNN sur le jeu de données Fashion MNIST.
  - Sauvegarde le modèle final dans le répertoire `src/models`.
  - Intègre MLflow pour suivre les performances d'entraînement.

- **`test_api.py`** :
  - Script permettant de tester les prédictions via l'API FastAPI.
  - Utile pour valider l'intégration backend et le fonctionnement des endpoints.

- **`test_img.py`** :
  - Affiche un échantillon des images générées dans le dossier `generated_images/`.
  - Vérifie que les images sont correctement créées et prêtes pour les tests.

### Scripts de modèles

- **`cnn.py`** :
  - Implémente un modèle CNN avec des couches de convolution, de pooling, et de dropout pour éviter l'overfitting.
  - Supporte l'entraînement avec augmentation de données et le suivi des performances avec MLflow.

- **`ml_classique.py`** :
  - Contient des implémentations pour les modèles classiques comme SVM et Random Forest.

- **`reseau_dense.py`** :
  - Implémente un réseau dense (MLP) pour comparer ses performances avec le CNN.

### Notebooks

- **`01_exploration.ipynb`** : Analyse exploratoire des données Fashion MNIST.
- **`02_model_ml_classique.ipynb`** : Implémentation et comparaison des modèles de machine learning classique.
- **`03_model_reseau_dense.ipynb`** : Implémentation du réseau dense.
- **`04_model_cnn.ipynb`** : Entraînement du modèle CNN.
- **`05_mlflow_tracking.ipynb`** : Suivi des expériences avec MLflow.

## Instructions d'installation

1. Clonez ce dépôt :
   ```bash
   git clone <url-du-repo>
   cd PROJET_MNIST_FASHION
   ```
2. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Sous Windows : .venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Instructions d'exécution

### Entraînement du modèle CNN

1. Lancez le script d'entraînement :
   ```bash
   python training_cnn.py
   ```
2. Le modèle final sera sauvegardé dans `src/models/cnn_model.h5`.

### Tester avec Streamlit

1. Lancez l'application Streamlit :
   ```bash
   streamlit run app.py
   ```
2. Ouvrez l'interface dans votre navigateur (lien fourni dans le terminal).

### Tester l'API

1. Lancez l'API FastAPI :
   ```bash
   uvicorn app:app --reload
   ```
2. Accédez à la documentation Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

3. Testez les prédictions via `test_api.py` :
   ```bash
   python test_api.py
   ```

## Ressources supplémentaires

- Dataset : [Fashion MNIST sur Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
- Documentation Streamlit : [https://docs.streamlit.io](https://docs.streamlit.io)
- Documentation MLflow : [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)