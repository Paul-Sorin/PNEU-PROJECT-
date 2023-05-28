import os

# Chemin vers le dossier "Tire_Textures"
base_dir = "/Users/paul/Downloads/Tire_Textures"

# Chemin vers le dossier d'entraînement
training_dir = os.path.join(base_dir, "training_data")

# Chemin vers le dossier de test
testing_dir = os.path.join(base_dir, "testing_data")

# Fonction pour parcourir les images dans un répertoire et ses sous-répertoires
def process_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                # Faites quelque chose avec l'image, comme l'ajouter à votre ensemble de données d'entraînement

# Parcourir les images du dossier d'entraînement
for cls in ["cracked", "normal"]:
    cls_dir = os.path.join(training_dir, cls)
    if os.path.exists(cls_dir):
        process_images(cls_dir)

# Parcourir les images du dossier de test
for cls in ["cracked", "normal"]:
    cls_dir = os.path.join(testing_dir, cls)
    if os.path.exists(cls_dir):
        process_images(cls_dir)
