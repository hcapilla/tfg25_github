import os

# Ruta al dataset
dataset_path = "C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/model_dataset v2 - Detection"
yaml_output_path = os.path.join(dataset_path, "dataset.yaml")

# Clases de tu dataset
classes = [
    "HVAC", "actuator", "agitator", "air conditioning", "architectural", "arrows", "bin", "blender",
    "blower", "boiler", "car", "centrifuge", "chiller", "chucker", "coating", "collector", "column",
    "compressor", "computer", "computer key", "container", "controller", "conveyor", "crystallizer",
    "deaerator", "detector", "digester", "drill", "dryer", "duct", "electrical", "equipment", "evaporator",
    "fan", "feeder", "filter", "finishing", "flow", "general", "generator", "grinder", "heater", "holder",
    "hopper", "indicator", "machining", "meter", "mill", "minery", "misc", "mixer", "motor", "oven",
    "packaging", "pipe", "press", "processor", "pump", "reactor", "regulator", "sensor", "separator",
    "signal", "silo", "stripping", "switch", "tank", "totalizer", "tower", "trap", "tube", "turbine",
    "vacuum", "valve", "vessel", "washer"
]

# Generar contenido del archivo YAML
yaml_content = f"""
# dataset.yaml generado automáticamente
path: {dataset_path.replace("\\", "/")}
train: images/train
val: images/val
nc: {len(classes)}
names:
{os.linesep.join(['  - ' + cls for cls in classes])}
"""

# Guardar el archivo YAML
with open(yaml_output_path, "w") as yaml_file:
    yaml_file.write(yaml_content)

print(f"Archivo YAML generado en: {yaml_output_path}")
