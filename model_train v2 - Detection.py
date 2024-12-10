from ultralytics import YOLO

def train_model():
    # Configuración
    dataset_yaml_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/model_dataset v2 - Detection/dataset.yaml"  # Ruta al archivo dataset.yaml
    model_checkpoint = "yolov8n.pt"  # Modelo YOLO preentrenado (nano). Cambia a 'yolov8s.pt', 'yolov8m.pt', etc., según tu necesidad.
    output_project_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Amber Detection Model"  # Carpeta para guardar resultados
    experiment_name = "Amber"  # Nombre del experimento

    # Entrenamiento
    model = YOLO(model_checkpoint)  # Cargar modelo preentrenado

    # Ejecutar el entrenamiento
    model.train(
        data=dataset_yaml_path,      # Ruta al archivo YAML del dataset
        epochs=100,                  # Número de épocas (ajústalo según tu dataset)
        imgsz=640,                   # Tamaño de las imágenes (640x640 recomendado)
        batch=32,                    # Tamaño del lote (ajusta según la VRAM de tu GPU)
        lr0=0.01,                    # Tasa de aprendizaje inicial
        device=0,                    # Dispositivo (0 para GPU, 'cpu' para usar CPU)
        project=output_project_path, # Carpeta donde guardar resultados
        name=experiment_name         # Nombre del experimento
    )

if __name__ == "__main__":
    train_model()
