import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Chemin vers le dossier contenant les photos de pneus
data_dir = "/Users/paul/Downloads/Tire_Textures"

# Liste des classes (pneus, matelas, palettes, sacs poubelles)
classes = ["pneus", "matelas", "palettes", "sacs poubelles"]

# Prétraitement des données
data = []
labels = []

# Parcours des classes
for i, cls in enumerate(classes):
    # Chemin vers le dossier de la classe
    cls_dir = os.path.join(data_dir, cls)
    # Parcours des images de la classe
    for img_file in os.listdir(cls_dir):
        # Chargement de l'image en niveaux de gris
        img_path = os.path.join(cls_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Redimensionnement de l'image à une taille fixe (par exemple 224x224)
        img = cv2.resize(img, (224, 224))
        # Ajout de l'image et de l'étiquette aux listes
        data.append(img)
        labels.append(cls)

# Conversion des listes en tableaux numpy
data = np.array(data)
labels = np.array(labels)

# Encodage des étiquettes
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Séparation des données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Chargement du modèle MobileNetV2 pré-entraîné
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ajout de nouvelles couches de classification au sommet du modèle
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Définition du modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Entraînement du modèle
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Évaluation du modèle sur l'ensemble de test
loss, accuracy = model.evaluate(test_data, test_labels)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Sauvegarde du modèle entraîné
model.save("trained_model.h5")
