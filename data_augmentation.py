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
                    # Generar y guardar la imagen con volteo horizontal
                    flipped_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_name_horizontal = os.path.splitext(image_file)[0] + "_2" + os.path.splitext(image_file)[1]
                    flipped_horizontal.save(os.path.join(class_path, new_name_horizontal))

                    # Generar y guardar la imagen con volteo vertical
                    flipped_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)
                    new_name_vertical = os.path.splitext(image_file)[0] + "_3" + os.path.splitext(image_file)[1]
                    flipped_vertical.save(os.path.join(class_path, new_name_vertical))

                    # Generar y guardar la imagen con volteo horizontal y vertical
                    flipped_both = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                    new_name_both = os.path.splitext(image_file)[0] + "_4" + os.path.splitext(image_file)[1]
                    flipped_both.save(os.path.join(class_path, new_name_both))
