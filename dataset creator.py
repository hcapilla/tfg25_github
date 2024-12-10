import os
import shutil
from sklearn.model_selection import train_test_split

# Ruta del dataset original
dataset_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Symbol dataset - Order"
output_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/model_dataset"

# Crear carpetas de salida
os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test"), exist_ok=True)

# Recorremos las clases
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        # Crear subcarpetas para cada clase en train, val y test
        os.makedirs(os.path.join(output_path, "train", class_folder), exist_ok=True)
        os.makedirs(os.path.join(output_path, "val", class_folder), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", class_folder), exist_ok=True)

        # Obtener todas las imágenes de la clase
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

        # Dividir en train, val y test
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.33, random_state=42)  # ~20% val, ~10% test

        # Copiar imágenes
        for img in train:
            shutil.copy(img, os.path.join(output_path, "train", class_folder))
        for img in val:
            shutil.copy(img, os.path.join(output_path, "val", class_folder))
        for img in test:
            shutil.copy(img, os.path.join(output_path, "test", class_folder))
