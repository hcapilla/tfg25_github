from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = YOLO("C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Amber Detection Model/Amber4/weights/best.pt")

# Ruta de la imagen original
image_path = "input/Sinoptico2.png"

# Realizar predicción
results = model.predict(source=image_path, conf=0.05)

# Cargar la imagen original
original_image = cv2.imread(image_path)

# Procesar las detecciones
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    confs = result.boxes.conf.cpu().numpy()  # Confidences
    classes = result.boxes.cls.cpu().numpy()  # Class IDs
    names = model.names  # Obtener nombres de las clases

    # Dibujar las bounding boxes en la imagen original
    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {conf:.2f}"  # Nombre de clase y confianza
        color = (0, 255, 0)  # Color verde para las cajas
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)  # Dibujar la caja
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Etiqueta

# Mostrar la imagen con Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para mostrar correctamente
plt.axis('off')
plt.title("Resultados de la Detección")
plt.show()

# Guardar la imagen procesada
output_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/tfg25_github/output/Sinoptico2_detected.png"
cv2.imwrite(output_path, original_image)
print(f"Imagen procesada guardada en: {output_path}")