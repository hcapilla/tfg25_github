# Global
import argparse
import os
from collections import Counter
import random

# SVG 
import svgwrite as svg
from xml.etree import ElementTree as ET

# OCR
from PIL import Image, ImageDraw
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# Template matching
import cv2
import numpy as np

def ReSin_config():
    parser = argparse.ArgumentParser(description="ReSin configuration")

    # GLOBAL.
    parser.add_argument('--workspace', type=str, default='factory1/', help='Current factory workspace')
    parser.add_argument('--input', type=str, default='input/', help='Input image folder')

    # OCR
    parser.add_argument('--ocr_output', type=str, default='output_ocr/', help='Output OCR image folder')
    parser.add_argument('--ocr_language', type=str, default='en', help='Languages')
    parser.add_argument('--ocr_confidence_threshold', type=float, default='0.7', help='OCR Confidence threshold')

    # TEMPLATE MATCHING
    parser.add_argument('--template_library', type=str, default='library/', help='Template folder')
    parser.add_argument('--template_output', type=str, default='output_template/', help='Output Post-TemplateMatching folder')


    arguments = parser.parse_args()

    s_workspace_path = arguments.workspace
    s_input_path = arguments.input

    s_ocr_output_path = arguments.ocr_output
    s_ocr_language = arguments.ocr_language.split(',')
    s_ocr_confidence_threshold = arguments.ocr_confidence_threshold

    s_template_library = arguments.template_library
    s_template_output_path = arguments.template_output

    s_input_path, s_ocr_output_path, s_template_library, s_template_output_path = CreatePaths(
        s_workspace_path, s_input_path, s_ocr_output_path, s_template_library, s_template_output_path
    )

    return s_input_path, s_workspace_path, s_ocr_output_path, s_ocr_language, s_ocr_confidence_threshold, s_template_library, s_template_output_path

def TEST_showConfidence(detection):
    text_lines = detection[0].text_lines
    text_lines_ordenado = sorted(text_lines, key=lambda line: line.confidence)

    for item in text_lines_ordenado:
        print(f"Text: {item.text}, Confidence: {item.confidence}")

def TEST_drawOriginalPNGBoundingBoxes(image, text_line, draw):
    x_min, y_min = text_line.bbox[0], text_line.bbox[1]
    x_max, y_max = text_line.bbox[2], text_line.bbox[3]

    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    cropped_area = image.crop((x_min, y_min, x_max, y_max))
    colors = cropped_area.getcolors(cropped_area.size[0] * cropped_area.size[1])
    if colors:
        most_common_color = max(colors, key=lambda x: x[0])[1]
        print(f"Color más común: {most_common_color}")

def TEST_drawSVGBoundingBoxes(text_line, output_svg):
    if text_line.confidence < 0.7:
        boundingBox_color = "red" 
    elif text_line.confidence < 0.82: 
        boundingBox_color = "yellow"
    else:
        boundingBox_color = "blue"
    
    output_svg.add(output_svg.polygon(
        points=text_line.polygon,
        fill="none",
        stroke=boundingBox_color,
        stroke_width=2
    ))
    
def CreatePaths(workspace, input_path, ocr_output_path, template_library, template_output_path):
    # Añadir el workspace como prefijo a los paths
    input_path = os.path.join(workspace, input_path)
    ocr_output_path = os.path.join(workspace, ocr_output_path)
    template_library = os.path.join(workspace, template_library)
    template_output_path = os.path.join(workspace, template_output_path)

    # Crear los directorios si no existen
    for path in [input_path, ocr_output_path, template_library, template_output_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    return input_path, ocr_output_path, template_library, template_output_path

def deleteByThreshold(detection, ocr_confidence_threshold):
    def is_valid_line(line):
        # Verifica que la línea tenga confianza suficiente
        if line.confidence < ocr_confidence_threshold:
            return False

        # Verifica que la línea no tenga solo un carácter
        if len(line.text) == 1:
            return False

        # Verifica que la línea no tenga solo dos caracteres
        if len(line.text) == 2:
            return False

        # Verifica que la línea no esté compuesta completamente por el mismo carácter
        if len(set(line.text)) == 1:
            return False

        # Elimina todo lo que haya después de "(" si no contiene ")"
        if "(" in line.text and ")" not in line.text:
            line.text = line.text.split("(")[0]

        # Elimina todo lo que haya después de "[" si no contiene "]"
        if "[" in line.text and "]" not in line.text:
            line.text = line.text.split("[")[0]

        return True

    # Filtra las líneas según las condiciones
    detection[0].text_lines = [line for line in detection[0].text_lines if is_valid_line(line)]

    return detection

def createOutputName(output_dir, base_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_name_svg = f"{base_name}.svg"
    output_name_png = f"{base_name}.png"

    output_path_svg = os.path.join(output_dir, output_name_svg)
    output_path_png = os.path.join(output_dir, output_name_png)
    
    if not os.path.exists(output_path_svg) and not os.path.exists(output_path_png):
        return output_path_svg, output_path_png
    
    counter = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}({counter}).svg")) or os.path.exists(os.path.join(output_dir, f"{base_name}({counter}).png")):
        counter += 1

    output_path_svg = os.path.join(output_dir, f"{base_name}({counter}).svg")
    output_path_png = os.path.join(output_dir, f"{base_name}({counter}).png")
    
    return output_path_svg, output_path_png

def embed_svg(output_svg, svg_file, x, y, width, height, rotation):
    def parse_dimension(value):
        """Convierte dimensiones como '5.3646in' o '120px' a un número flotante."""
        if "in" in value:
            return float(value.replace("in", "")) * 96  # 1in = 96px
        elif "cm" in value:
            return float(value.replace("cm", "")) * 37.7952755906  # 1cm = 37.79px
        elif "mm" in value:
            return float(value.replace("mm", "")) * 3.77952755906  # 1mm = 3.779px
        elif "px" in value:
            return float(value.replace("px", ""))
        else:
            return float(value)  # Asume que es un valor numérico puro

    # Leer y procesar el archivo SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Extraer y convertir las dimensiones
    svg_width = parse_dimension(root.attrib.get("width", "100px"))
    svg_height = parse_dimension(root.attrib.get("height", "100px"))

    # Calcular escala
    scale_x = width / svg_width
    scale_y = height / svg_height
    transform = f"translate({x},{y}) scale({scale_x},{scale_y})"
    if rotation != 0:
        cx, cy = x + width / 2, y + height / 2  # Centro para rotar
        transform += f" rotate({rotation},{cx},{cy})"

    # Cargar el contenido SVG externo como texto
    with open(svg_file, "r") as f:
        svg_content = f.read()

    # Crear un grupo y añadir el contenido
    group = output_svg.g(transform=transform)
    group.add(output_svg.text(svg_content, insert=(0, 0)))
    output_svg.add(group)

# OCR
def text_detection(input_path, output_path, language, ocr_confidence_threshold):
    output_images = []

    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_path, input_image)

            with Image.open(image_path) as image:
                output_path_SVG, output_path_PNG = createOutputName(output_path, os.path.splitext(input_image)[0])

                # OCR
                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()

                detection = run_ocr([image], [language], det_model, det_processor, rec_model, rec_processor)
                detection = deleteByThreshold(detection, ocr_confidence_threshold)

                # Output SVG and image attributes
                image_width = image._size[0]
                image_height = image._size[1]
                output_svg = svg.Drawing(output_path_SVG, size=(image_width, image_height))
                draw = ImageDraw.Draw(image)

                for text_line in detection[0].text_lines:
                    # Extraer las coordenadas de la Bounding Box
                    x_min, y_min = int(text_line.bbox[0]), int(text_line.bbox[1])
                    x_max, y_max = int(text_line.bbox[2]), int(text_line.bbox[3])

                    # Calcular el color más común en el área de la Bounding Box
                    cropped_area = image.crop((x_min, y_min, x_max, y_max))
                    colors = cropped_area.getcolors(cropped_area.size[0] * cropped_area.size[1])
                    if colors:
                        most_common_color = max(colors, key=lambda x: x[0])[1]

                        # Sustituir el contenido de la Bounding Box por el color predominante
                        for x in range(x_min, x_max):
                            for y in range(y_min, y_max):
                                image.putpixel((x, y), most_common_color)

                    # TEST_drawOriginalPNGBoundingBoxes(image, text_line, draw)  # DEBUG

                    # SVG
                    text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                    text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                    font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.9

                    # TEST_drawSVGBoundingBoxes(text_line, output_svg)

                    output_svg.add(output_svg.text(text_line.text,
                                                   insert=(text_x, text_y),
                                                   text_anchor="middle",
                                                   alignment_baseline="middle",
                                                   font_size=font_height,
                                                   font_weight="bold",
                                                   font_family="Tahoma"))

                # Guardar la imagen procesada con las Bounding Boxes sustituidas
                image.save(output_path_PNG)
                output_svg.save()
                output_images.append([output_path_PNG, output_path_SVG])

    print("OCR text detection completado.")
    return output_images

# Template matching
def calculate_iou(box1, box2):
    """
    Calcula el IoU (Intersection over Union) entre dos bounding boxes.
    Las bounding boxes se representan como (x, y, ancho, alto).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordenadas de la intersección
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Área de intersección
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Áreas de las cajas
    area1 = w1 * h1
    area2 = w2 * h2

    # Área de unión
    union_area = area1 + area2 - inter_area

    # Evitar divisiones por cero
    if union_area == 0:
        return 0

    # IoU
    return inter_area / union_area

def TemplateMatching(processed_images, output_path, library_path, final_svg_path, iou_threshold=0.5, template_threshold=0.55):
    if not processed_images:
        print("No hay imágenes procesadas para analizar.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Crear el SVG final
    final_svg = svg.Drawing(final_svg_path)

    # Colores únicos para las clases
    class_colors = {}

    # Iterar sobre las imágenes procesadas
    for image_pair in processed_images:
        if not isinstance(image_pair, list) or len(image_pair) < 2:
            print(f"Advertencia: Estructura incorrecta en processed_images: {image_pair}.")
            continue

        png_path, svg_path = image_pair

        if not os.path.exists(png_path):
            print(f"Advertencia: La imagen PNG {png_path} no existe.")
            continue

        input_image = cv2.imread(png_path)
        if input_image is None:
            print(f"Advertencia: No se pudo cargar la imagen {png_path}. Se omitirá.")
            continue

        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Realizar detecciones iniciales
        detections = []
        for class_name in os.listdir(library_path):
            class_folder = os.path.join(library_path, class_name)
            templates_folder = os.path.join(class_folder, "templates")

            if not os.path.exists(templates_folder):
                continue

            print(f"Accediendo a la carpeta de templates: {templates_folder}")

            for template_file in os.listdir(templates_folder):
                template_path = os.path.join(templates_folder, template_file)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

                if template is None:
                    continue

                for rotation_angle in [0, 90, 180, 270]:
                    rotated_template = template
                    if rotation_angle != 0:
                        # Rotar el template en ángulos de 90°, 180°, y 270°
                        rotated_template = cv2.rotate(template, {
                            90: cv2.ROTATE_90_CLOCKWISE,
                            180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE
                        }[rotation_angle])

                    result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= template_threshold)

                    for point in zip(*locations[::-1]):
                        top_left = point
                        bbox = (top_left[0], top_left[1], rotated_template.shape[1], rotated_template.shape[0])

                        overlaps = False
                        for existing_bbox in detections:
                            if calculate_iou(bbox, existing_bbox[:4]) > iou_threshold:
                                overlaps = True
                                break

                        if not overlaps:
                            detections.append((bbox[0], bbox[1], bbox[2], bbox[3], rotation_angle, class_name))

        for x, y, w, h, rotation, class_name in detections:
            if class_name not in class_colors:
                class_colors[class_name] = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

            color = tuple(int(c) for c in class_colors[class_name].strip("rgb()").split(","))
            rotation_display = rotation if rotation in [90, 180, 270] else 0
            cv2.rectangle(input_image, (x, y), (x + w, y + h), color, 1)
            cv2.putText(input_image, f"{class_name} ({rotation_display})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        output_image_path = os.path.join(output_path, os.path.basename(png_path))
        cv2.imwrite(output_image_path, input_image)
        print(f"Imagen inicial guardada con bounding boxes: {output_image_path}")

        while True:
            roi = cv2.selectROI("Selecciona la plantilla", input_image, showCrosshair=True)
            cv2.destroyWindow("Selecciona la plantilla")

            # Verificar si el usuario presionó ESC o seleccionó una región inválida
            if roi[2] == 0 or roi[3] == 0:
                print("Selección terminada o región inválida.")
                break

            # Extraer la plantilla seleccionada
            template = gray_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            if template.size == 0:
                print("Error: No se seleccionó una plantilla válida. Intente nuevamente.")
                continue

            # Pedir al usuario la clase del elemento
            class_name = input("Introduce la clase a la que pertenece el elemento seleccionado: ")
            if not class_name:
                print("Clase no válida. Intente nuevamente.")
                continue

            # Crear carpeta para la clase si no existe en la librería
            class_folder = os.path.join(library_path, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            templates_folder = os.path.join(class_folder, "templates")
            if not os.path.exists(templates_folder):
                os.makedirs(templates_folder)

            # Guardar el template original y variaciones
            iteration = 0
            height, width = template.shape[:2]

            # Aumentar dimensiones iterativamente
            while width < 2 * template.shape[1] and height < 2 * template.shape[0]:
                if iteration % 2 == 0:
                    width += 2
                else:
                    height += 2

                resized_template = cv2.resize(template, (width, height))
                cv2.imwrite(os.path.join(templates_folder, f"template_increase_{iteration}.png"), resized_template)
                iteration += 1

            # Reducir dimensiones iterativamente
            width, height = template.shape[1], template.shape[0]
            iteration = 0

            while width > template.shape[1] // 2 and height > template.shape[0] // 2:
                if iteration % 2 == 0:
                    width -= 2
                else:
                    height -= 2

                resized_template = cv2.resize(template, (max(width, 1), max(height, 1)))
                cv2.imwrite(os.path.join(templates_folder, f"template_decrease_{iteration}.png"), resized_template)
                iteration += 1

            print(f"Variaciones de plantilla guardadas en la carpeta 'templates' para la clase '{class_name}'.")

            # Aplicar template matching para la clase seleccionada
            detections = []
            for template_file in os.listdir(templates_folder):
                template_path = os.path.join(templates_folder, template_file)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

                if template is None:
                    continue

                for rotation_angle in [0, 90, 180, 270]:
                    rotated_template = template
                    if rotation_angle != 0:
                        rotated_template = cv2.rotate(template, {
                            90: cv2.ROTATE_90_CLOCKWISE,
                            180: cv2.ROTATE_180,
                            270: cv2.ROTATE_90_COUNTERCLOCKWISE
                        }[rotation_angle])

                    result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= template_threshold)

                    for point in zip(*locations[::-1]):
                        top_left = point
                        bbox = (top_left[0], top_left[1], rotated_template.shape[1], rotated_template.shape[0])

                        overlaps = False
                        for existing_bbox in detections:
                            if calculate_iou(bbox, existing_bbox[:4]) > iou_threshold:
                                overlaps = True
                                break

                        if not overlaps:
                            detections.append((bbox[0], bbox[1], bbox[2], bbox[3], rotation_angle, class_name))

            for x, y, w, h, rotation, class_name in detections:
                if class_name not in class_colors:
                    class_colors[class_name] = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

                color = tuple(int(c) for c in class_colors[class_name].strip("rgb()").split(","))
                rotation_display = rotation if rotation in [90, 180, 270] else 0
                cv2.rectangle(input_image, (x, y), (x + w, y + h), color, 1)
                cv2.putText(input_image, f"{class_name} ({rotation_display})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

            output_image_path = os.path.join(output_path, os.path.basename(png_path))
            cv2.imwrite(output_image_path, input_image)
            print(f"Imagen actualizada guardada con bounding boxes: {output_image_path}")

    # Guardar el SVG final
    final_svg.save()
    print("SVG final generado con éxito.")


def main():
    # OCR arguments
    input_path, workspace, ocr_output_path, ocr_language, ocr_confidence_threshold, template_library, template_output_path = ReSin_config()

    # OCR
    processed_images = text_detection(input_path, ocr_output_path, ocr_language, ocr_confidence_threshold)

    # Template matching
    final_svg_path = os.path.join(template_output_path, "final_diagram.svg")
    TemplateMatching(processed_images, template_output_path, template_library, final_svg_path)

    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()