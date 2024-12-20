## Support de Discussion pour le Projet MNIST Fashion

### 1. **Objectifs du Projet**
- **Comprendre et exploiter les architectures de réseaux de neurones convolutifs (CNN)** pour classifier les images du dataset Fashion MNIST.
- Comparer les performances des CNN avec des approches de machine learning classique (SVM, Random Forest) et des réseaux de neurones denses.
- Mettre en place un suivi des entraînements avec MLflow pour une meilleure traçabilité des expériences.
- Construire une API pour déployer le modèle et tester les prédictions avec des images.
- Créer une interface utilisateur avec Streamlit pour rendre le projet interactif.

### 2. **Structure du Projet**
- **`src/`**
  - `data/`: Chargement et prétraitement des données Fashion MNIST.
  - `models/`: Contient les modèles implémentés (​ML classique, dense, CNN).
  - `visualization/`: Outils pour visualiser les données et les résultats des modèles.
  - `app.py`: API Streamlit pour tester les prédictions.
- **`notebooks/`**
  - Exploration, entraînement des modèles, et suivi des expériences avec MLflow.
- **`mlruns/`**
  - Logs des expériences MLflow.
- **`generated_images/`**
  - Images générées pour tester le modèle.

### 3. **Cheminement Suivi**
#### **Étape 1 : Exploration des Données**
- Inspection des dimensions, des classes et de la distribution des données.
- Visualisation des échantillons pour comprendre la nature des images (grayscale 28x28).
- Analyse statistique pour identifier des anomalies ou biais potentiels.

#### **Étape 2 : Implémentation des Modèles**
- **ML Classique :**
  - Implémentation d'un SVM et d'un Random Forest.
  - Préparation des données avec aplatissement (28x28 → 784).
  - Résultats limités (≈ 85% de précision).
- **Réseaux de Neurones Denses :**
  - Architecture avec 2 couches denses et une sortie softmax.
  - Résultats supérieurs au ML classique (≈ 88-89% de précision).
- **CNN :**
  - Deux convolutions suivies de max-pooling et de dropout pour éviter l'overfitting.
  - Optimisation des hyperparamètres (epochs, batch size).
  - Précision atteinte : ≈ 90-92%.

#### **Étape 3 : Suivi des Expériences avec MLflow**
- Enregistrement des hyperparamètres (epochs, taux d'apprentissage).
- Suivi des métriques : `accuracy`, `loss` (entraînement/validation).
- Comparaison des performances entre les approches ML classique, dense, et CNN.

#### **Étape 4 : Tests avec des Images Personnalisées**
- Création d'images avec `create_img.py`.
- Tests sur des images générées (mélange de succès et d'échecs).
- Validation des performances sur les images du dataset Fashion MNIST (9/10 tests réussis).

#### **Étape 5 : Déploiement et Interaction**
- API avec Streamlit pour permettre aux utilisateurs de :
  - Obtenir les prédictions en temps réel.

### 4. **Éléments Clés pour les Questions**

#### **Pourquoi MLflow ?**
- Pour comparer objectivement les performances des modèles.
- Faciliter le suivi des différentes expériences.
- Gérer l'historique des hyperparamètres et des métriques.

#### **Pourquoi les CNN surpassent-ils les approches classiques ?**
- Les CNN capturent les dépendances spatiales dans les images grâce aux convolutions.
- Ils sont mieux adaptés pour traiter les données en 2D sans les aplatir.

#### **Pourquoi les résultats peuvent différer sur des images personnalisées ?**
- Les données générées manquent de réalisme ou de complexité comparées au dataset Fashion MNIST.
- Les images générées peuvent être biaisées par rapport à l'entraînement.

#### **Comment expliquer les graphes d'entraînement ?**
- **Loss en diminution régulière :** Montre que le modèle apprend efficacement.
- **Validation stable :** Indique que le modèle généralise bien, avec un léger risque d'overfitting contrôlé par le dropout.

#### **Quels sont les défis rencontrés ?**
- Préparer des images de test cohérentes avec le dataset.
- Éviter le sur-ajustement avec une architecture simple mais efficace.
- Gérer des prédictions incohérentes sur des images artificielles.

### 5. **Axes d'Amélioration**
- **Data Augmentation :**
  - Rotation, zoom, translation pour augmenter la diversité des données d'entraînement.
- **Hyperparameter Tuning :**
  - Ajuster le taux d'apprentissage, la taille des batchs, et les filtres convolutifs.
- **Ensemble Learning :**
  - Combiner les prédictions de plusieurs modèles pour améliorer la robustesse.

### 6. **Préparation pour les Questions**
- **Maîtriser les chiffres :** Résultats de précision et de perte.
- **Comprendre les choix :** Pourquoi une approche plutôt qu'une autre.
- **Expliquer les erreurs :** Identifier les limitations des données et des modèles.
- **Suggérer des améliorations :** Montrer une compréhension critique et proactive du projet.

### 7. **Annexes**
- **Ressources utilisées :**
  - Dataset Fashion MNIST.
  - TensorFlow/Keras pour les modèles.
  - Streamlit pour l'interface utilisateur.
  - MLflow pour le suivi des expériences.
- **Commandes Clés :**
  - Démarrer l'API : `streamlit run app.py`.
  - Tester l'API : `python test_api.py`.
  - Générer des images : `python create_img.py`.