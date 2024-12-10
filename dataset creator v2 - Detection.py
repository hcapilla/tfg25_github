import os
import random
import shutil
from PIL import Image

# Configuración
dataset_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Symbol dataset - Order"  # Ruta a tu dataset original
output_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/model_dataset v2 - Detection"  # Ruta al dataset generado
train_ratio = 0.8  # Proporción de entrenamiento

# Crear carpetas para el dataset organizado
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_path, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_path, f"labels/{split}"), exist_ok=True)

# Obtener las clases y procesar cada carpeta
classes = sorted(os.listdir(dataset_path))
class_map = {class_name: idx for idx, class_name in enumerate(classes)}
print("Clases detectadas:", class_map)

for class_name, class_id in class_map.items():
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split, split_images in splits.items():
        for img_name in split_images:
            img_src_path = os.path.join(class_dir, img_name)
            img_dst_path = os.path.join(output_path, f"images/{split}", f"{class_name}_{img_name}")
            label_dst_path = os.path.join(output_path, f"labels/{split}", f"{class_name}_{os.path.splitext(img_name)[0]}.txt")

            # Copiar la imagen al destino
            shutil.copy(img_src_path, img_dst_path)

            # Generar etiqueta YOLO
            with Image.open(img_src_path) as img:
                width, height = img.size
            with open(label_dst_path, "w") as label_file:
                # Bounding box ocupa toda la imagen: clase_id 0.5 0.5 1.0 1.0
                label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print(f"Dataset reorganizado guardado en: {output_path}")
