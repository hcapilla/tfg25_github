import os
from PIL import Image

# Ruta a la carpeta del dataset
dataset_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Symbol dataset - Order"

# Recorremos cada subcarpeta en la carpeta principal
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    # Verificamos que sea una carpeta
    if os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)

            # Verificamos que sea una imagen válida
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(image_path) as img:
                    # Generar y guardar la imagen en escala de grises
                    grayscale = img.convert("L")
                    new_name_grayscale = os.path.splitext(image_file)[0] + "_gray" + os.path.splitext(image_file)[1]
                    grayscale.save(os.path.join(class_path, new_name_grayscale))

print("Conversión a escala de grises completada.")
