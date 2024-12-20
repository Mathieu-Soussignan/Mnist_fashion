import numpy as np
from PIL import Image, ImageDraw
import os

# Dossier pour sauvegarder les images générées
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Classes Fashion MNIST
classes = {
    0: "T-shirt/top",
    1: "Pantalon",
    2: "Pull",
    3: "Robe",
    4: "Manteau",
    5: "Sandale",
    6: "Chemise",
    7: "Basket",
    8: "Sac",
    9: "Botte/chaussure"
}

def create_image(class_id, output_dir):
    """
    Crée une image 28x28 simulant une classe de Fashion MNIST.

    Args:
        class_id (int): ID de la classe (0-9).
        output_dir (str): Dossier de sauvegarde des images.

    Returns:
        None
    """
    # Vérifie que la classe existe
    if class_id not in classes:
        raise ValueError(f"Classe {class_id} non valide. Choisissez entre 0 et 9.")

    # Créer une image vide (fond noir)
    image = Image.new("L", (28, 28), 0)  # 'L' pour niveaux de gris
    draw = ImageDraw.Draw(image)

    # Dessiner un motif simple basé sur la classe
    if class_id == 0:  # T-shirt/top
        draw.rectangle([6, 6, 22, 18], fill=255)  # Rectangle pour le haut
    elif class_id == 1:  # Pantalon
        draw.rectangle([10, 8, 18, 20], fill=255)  # Rectangle pour les jambes
    elif class_id == 2:  # Pull
        draw.rectangle([6, 6, 22, 22], outline=255, width=2)  # Rectangle avec contour
    elif class_id == 3:  # Robe
        draw.polygon([(14, 6), (6, 22), (22, 22)], fill=255)  # Triangle pour la robe
    elif class_id == 4:  # Manteau
        draw.rectangle([4, 4, 24, 24], outline=255, width=2)  # Contour large
    elif class_id == 5:  # Sandale
        draw.ellipse([10, 10, 18, 18], fill=255)  # Cercle pour une sandale
    elif class_id == 6:  # Chemise
        draw.rectangle([8, 6, 20, 18], fill=255)  # Rectangle pour le corps
        draw.rectangle([6, 4, 8, 6], fill=255)  # Manche gauche
        draw.rectangle([20, 4, 22, 6], fill=255)  # Manche droite
    elif class_id == 7:  # Basket
        draw.rectangle([8, 16, 20, 22], fill=255)  # Semelle
        draw.rectangle([10, 8, 18, 16], fill=255)  # Partie supérieure
    elif class_id == 8:  # Sac
        draw.rectangle([8, 8, 20, 20], fill=255)  # Sac
        draw.rectangle([10, 6, 18, 8], fill=255)  # Poignée
    elif class_id == 9:  # Botte/chaussure
        draw.rectangle([10, 14, 18, 22], fill=255)  # Bas de la botte
        draw.rectangle([12, 8, 16, 14], fill=255)  # Haut de la botte

    # Sauvegarder l'image
    class_name = classes[class_id]
    filename = os.path.join(output_dir, f"class_{class_id}_{class_name.replace('/', '_')}.png")
    image.save(filename)
    print(f"Image sauvegardée : {filename}")

# Générer des images pour toutes les classes
for class_id in classes.keys():
    create_image(class_id, output_dir)

print(f"Toutes les images ont été générées dans le dossier : {output_dir}")