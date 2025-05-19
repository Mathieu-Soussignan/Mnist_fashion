from tensorflow.keras.models import load_model

# Charger le modèle original (format .h5)
model = load_model("./src/models/cnn_model.h5")

# Sauvegarder au format Keras standard (.keras)
model.save("./src/models/cnn_model_converted.keras")

print("✅ Modèle converti et sauvegardé avec succès.")