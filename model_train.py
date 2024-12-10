from ultralytics import YOLO

if __name__ == "__main__":
    # Ruta a tu dataset organizado
    dataset_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/model_dataset"

    # Cargar modelo YOLO preentrenado para clasificación
    model = YOLO("yolov8n-cls.pt")  # Modelo pequeño, ideal para empezar

    # Entrenamiento
    model.train(
        data=dataset_path,   # Ruta al dataset organizado
        epochs=100,          # Número de épocas
        imgsz=224,           # Tamaño de las imágenes (224x224 recomendado)
        batch=32,            # Tamaño del lote (ajusta según la VRAM disponible)
        lr0=0.01,            # Tasa de aprendizaje inicial
        device=0,            # Asegura que se use la GPU
        project="TFG25_model",  # Nombre del proyecto (carpeta)
        name="Ambar model"      # Nombre del experimento
    )
