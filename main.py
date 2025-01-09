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
    parser.add_argument('--template_confidence_threshold', type=float, default='0.55', help='OCR Confidence threshold')
    parser.add_argument('--iou_confidence_threshold', type=float, default='0.5', help='OCR Confidence threshold')


    arguments = parser.parse_args()

    s_workspace_path = arguments.workspace
    s_input_path = arguments.input

    s_ocr_output_path = arguments.ocr_output
    s_ocr_language = arguments.ocr_language.split(',')
    s_ocr_confidence_threshold = arguments.ocr_confidence_threshold

    s_template_library = arguments.template_library
    s_template_output_path = arguments.template_output
    s_template_confidence_threshold = arguments.template_confidence_threshold
    s_iou_confidence_threshold = arguments.iou_confidence_threshold

    s_input_path, s_ocr_output_path, s_template_library, s_template_output_path = CreatePaths(
        s_workspace_path, s_input_path, s_ocr_output_path, s_template_library, s_template_output_path
    )

    return s_input_path, s_workspace_path, s_ocr_output_path, s_ocr_language, s_ocr_confidence_threshold, s_template_library, s_template_output_path, s_template_confidence_threshold, s_iou_confidence_threshold

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

def get_dominant_color(image_path):
    """
    Calcula el color predominante de una imagen.
    Si el color predominante es negro, retorna el segundo color más frecuente.

    Args:
        image_path (str): Ruta de la imagen.

    Returns:
        tuple: Color predominante en formato (R, G, B).
    """
    # Leer la imagen
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("No se pudo leer la imagen. Verifica la ruta.")

    # Convertir la imagen a formato RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplanar la matriz para obtener los colores en formato (R, G, B)
    pixels = image.reshape(-1, 3)

    # Contar la frecuencia de cada color
    pixel_counts = Counter(map(tuple, pixels))

    # Ordenar por frecuencia de mayor a menor
    sorted_colors = pixel_counts.most_common()

    # Buscar el color predominante que no sea negro
    for color, _ in sorted_colors:
        if color != (0, 0, 0):  # Ignorar el negro
            return color

    # Si todos los colores son negros, retornar negro
    return (0, 0, 0)

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

    output_path_svg = os.path.join(output_dir, f"{base_name}_{counter}.svg")
    output_path_png = os.path.join(output_dir, f"{base_name}_{counter}.png")
    
    return output_path_svg, output_path_png

def text_detection(input_path, output_path, language, ocr_confidence_threshold):
    output_images = []

    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_path, input_image)

            with Image.open(image_path) as image:
                output_path_SVG, output_path_PNG = createOutputName(output_path, os.path.splitext(input_image)[0])
                
                # Obtener el color predominante
                most_color = get_dominant_color(image_path)

                # OCR
                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()

                detection = run_ocr([image], [language], det_model, det_processor, rec_model, rec_processor)
                detection = deleteByThreshold(detection, ocr_confidence_threshold)

                # Output SVG and image attributes
                image_width = image._size[0]
                image_height = image._size[1]

                # Crear el SVG sin namespaces adicionales
                output_svg = svg.Drawing(output_path_SVG, size=(image_width, image_height), profile='full')
                output_svg['xmlns'] = 'http://www.w3.org/2000/svg'  # Namespace básico, sin prefijos

                # Convertir el color predominante (most_color) a formato hexadecimal para SVG
                most_color_hex = '#{:02x}{:02x}{:02x}'.format(*most_color)

                # Añadir un rectángulo que cubra todo el fondo del SVG con el color predominante
                output_svg.add(output_svg.rect(
                    insert=(0, 0),  # Posición inicial
                    size=(image_width, image_height),  # Tamaño del rectángulo (cubre todo el SVG)
                    fill=most_color_hex  # Color de fondo
                ))

                draw = ImageDraw.Draw(image)

                text_detections = []  # Lista para almacenar las detecciones de texto

                for text_line in detection[0].text_lines:
                    # Extraer las coordenadas de la Bounding Box
                    x_min, y_min = int(text_line.bbox[0]), int(text_line.bbox[1])
                    x_max, y_max = int(text_line.bbox[2]), int(text_line.bbox[3])

                    # Sustituir el contenido de la Bounding Box por el color predominante
                    for x in range(x_min, x_max):
                        for y in range(y_min, y_max):
                            image.putpixel((x, y), most_color)

                    # TEST_drawOriginalPNGBoundingBoxes(image, text_line, draw)  # DEBUG

                    # SVG
                    text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                    text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                    font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.9

                    # Ajustar el tamaño del texto basado en el rango del font_height
                    if 0 <= font_height <= 20:
                        font_size = 15
                    elif 20 < font_height <= 30:
                        font_size = 25
                    elif 30 < font_height <= 40:
                        font_size = 35
                    else:
                        font_size = int(font_height)  # Usar el tamaño original si es mayor a 40

                    # TEST_drawSVGBoundingBoxes(text_line, output_svg)  # DEBUG

                    output_svg.add(output_svg.text(
                        text_line.text,
                        insert=(text_x, text_y),
                        text_anchor="middle",
                        alignment_baseline="middle",
                        font_size=font_size,  # Usar el tamaño ajustado
                        font_weight="bold",
                        font_family="Tahoma"
                    ))

                    # Añadir la detección de texto con el flag
                    text_detections.append({
                        "bbox": [x_min, y_min, x_max, y_max],
                        "text": text_line.text,
                        "flag": "text"
                    })

                # Guardar la imagen procesada con las Bounding Boxes sustituidas
                image.save(output_path_PNG)
                output_svg.save()

                # Agregar las detecciones con el flag a la salida
                output_images.append([output_path_PNG, output_path_SVG, text_detections])

    print("OCR text detection completado.")
    return output_images



# Template matching
def calculate_iou(box1, box2, image, predominant_color):
    """
    Calcula el IoU (Intersection over Union) entre dos bounding boxes.
    Si las bounding boxes se superponen, prioriza aquella con MENOS píxeles
    del color predominante en la intersección.
    
    Args:
        box1: Tuple (x, y, ancho, alto) de la primera bounding box.
        box2: Tuple (x, y, ancho, alto) de la segunda bounding box.
        image: Imagen (array numpy) para analizar los píxeles.
        predominant_color: Tuple (B, G, R) con el color predominante.

    Returns:
        iou: El IoU entre las dos cajas. Si se prioriza una caja,
        se retorna el índice de la caja con más píxeles del color predominante.
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

    if inter_area == 0:
        # No hay superposición
        return 0

    # Extraer la región de intersección de la imagen
    intersection_region = image[yi1:yi2, xi1:xi2]

    # Calcular la cantidad de píxeles del color predominante en la región
    predominant_color_bgr = np.array(predominant_color, dtype=np.uint8)  # Asegurar formato
    mask = np.all(intersection_region == predominant_color_bgr, axis=-1)
    predominant_pixels_count = np.sum(mask)

    # Comparar el número de píxeles del color predominante en cada caja completa
    region1 = image[y1:y1 + h1, x1:x1 + w1]
    region2 = image[y2:y2 + h2, x2:x2 + w2]

    mask1 = np.all(region1 == predominant_color_bgr, axis=-1)
    mask2 = np.all(region2 == predominant_color_bgr, axis=-1)

    count1 = np.sum(mask1)
    count2 = np.sum(mask2)

    # Si las cajas tienen superposición, eliminar la que tenga más píxeles predominantes
    if count1 > count2:
        return 1  # Priorizar la eliminación de box1
    elif count2 > count1:
        return 2  # Priorizar la eliminación de box2

    # Si tienen igual cantidad de píxeles predominantes, calcular IoU como fallback
    area1 = w1 * h1
    area2 = w2 * h2

    # Área de unión
    union_area = area1 + area2 - inter_area

    # Evitar divisiones por cero
    if union_area == 0:
        return 0

    # IoU
    return inter_area / union_area


def TemplateMatching(processed_images, output_path, library_path, template_threshold, iou_threshold):
    if not processed_images:
        print("No hay imágenes procesadas para analizar.")
        return []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Colores únicos para las clases
    class_colors = {}

    # Lista de resultados para devolver
    output_images = []

    # Iterar sobre las imágenes procesadas
    for image_pair in processed_images:
        if not isinstance(image_pair, list) or len(image_pair) < 3:
            print(f"Advertencia: Estructura incorrecta en processed_images: {image_pair}.")
            continue

        png_path, svg_path, res = image_pair

        if not os.path.exists(png_path):
            print(f"Advertencia: La imagen PNG {png_path} no existe.")
            continue

        input_image = cv2.imread(png_path)
        if input_image is None:
            print(f"Advertencia: No se pudo cargar la imagen {png_path}. Se omitirá.")
            continue

        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Obtener el color predominante
        most_color = get_dominant_color(png_path)
        most_color = tuple(map(int, most_color[::-1]))  # Convertir a formato BGR

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
                            iou_or_priority = calculate_iou(bbox, existing_bbox[:4], input_image, most_color)

                            if isinstance(iou_or_priority, int):
                                if iou_or_priority == 1:
                                    overlaps = True
                                    break
                                elif iou_or_priority == 2:
                                    detections.remove(existing_bbox)
                                    break
                            elif iou_or_priority > iou_threshold:
                                overlaps = True
                                break

                        if not overlaps:
                            detections.append((bbox[0], bbox[1], bbox[2], bbox[3], rotation_angle, class_name))

        # Guardar las detecciones iniciales
        all_detections = detections.copy()

        while True:
            # Mostrar todas las bounding boxes acumuladas con clases y rotaciones en una copia temporal
            temp_image = input_image.copy()
            for x, y, w, h, rotation, class_name in all_detections:
                if class_name not in class_colors:
                    class_colors[class_name] = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

                color = tuple(int(c) for c in class_colors[class_name].strip("rgb()").split(","))
                rotation_display = rotation if rotation in [90, 180, 270] else 0
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(temp_image, f"{class_name} ({rotation_display})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

            # Selección de ROI
            roi = cv2.selectROI("Selecciona la plantilla", temp_image, showCrosshair=True)
            cv2.destroyWindow("Selecciona la plantilla")

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

            # Guardar el template original
            cv2.imwrite(os.path.join(templates_folder, "template.png"), template)

            # Agregar detección basada en la ROI seleccionada
            new_detection = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]), 0, class_name)

            # Verificar solapamientos antes de añadir la nueva detección
            overlaps = False
            for existing_bbox in all_detections:
                iou_or_priority = calculate_iou(new_detection[:4], existing_bbox[:4], input_image, most_color)

                if isinstance(iou_or_priority, int):
                    if iou_or_priority == 1:
                        overlaps = True
                        break
                    elif iou_or_priority == 2:
                        all_detections.remove(existing_bbox)
                        break
                elif iou_or_priority > iou_threshold:
                    overlaps = True
                    break

            if not overlaps:
                all_detections.append(new_detection)

        # Pintar bounding boxes finales con el color predominante
        for x, y, w, h, rotation, class_name in all_detections:
            cv2.rectangle(input_image, (x, y), (x + w, y + h), most_color, -1)

        # Guardar el resultado final
        output_template_png_path = os.path.join(output_path, os.path.basename(png_path))
        cv2.imwrite(output_template_png_path, input_image)
        print(f"Imagen actualizada con las nuevas detecciones encontradas: {output_template_png_path}")

        # Llamar a insert_detected_svgs para insertar los elementos SVG detectados
        insert_detected_svgs(all_detections, library_path, processed_images, output_path)

        # Agregar el flag justo antes de retornar el valor
        flagged_detections = [det + ("elemento",) for det in all_detections]

        # Agregar a la lista de resultados
        output_images.append([output_template_png_path, svg_path, flagged_detections])

    return output_images


def limpiar_namespaces(element):
    """Remueve los namespaces de los elementos XML."""
    for elem in element.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # Elimina el namespace

def parse_dimension(value):
    """Convierte dimensiones como '5.3646in', '120px', etc., a un número flotante en píxeles."""
    if "in" in value:
        return float(value.replace("in", "")) * 96  # 1 in = 96 px
    elif "cm" in value:
        return float(value.replace("cm", "")) * 37.7952755906  # 1 cm = 37.79 px
    elif "mm" in value:
        return float(value.replace("mm", "")) * 3.77952755906  # 1 mm = 3.779 px
    elif "px" in value:
        return float(value.replace("px", ""))
    else:
        return float(value)  # Asume un valor numérico puro si no hay unidad

def insert_detected_svgs(detections, library_path, processed_images, output_directory):
    """
    Inserta elementos SVG detectados en los archivos SVG correspondientes.

    Args:
        processed_images (list): Lista con pares de rutas [(png_path, svg_path), ...].
        detections (list): Lista de detecciones en formato [(x, y, w, h, rotation, class_name), ...].
        library_path (str): Ruta donde están los SVG de las clases detectadas (en library/general).
        output_directory (str): Directorio para guardar los archivos SVG actualizados.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for png_path, svg_path, res2 in processed_images:
        if not os.path.exists(svg_path):
            print(f"El archivo SVG base '{svg_path}' no existe. Se omitirá.")
            continue

        # Cargar el archivo SVG base
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # Limpiar namespaces del archivo base
            limpiar_namespaces(root)
        except Exception as e:
            print(f"Error al cargar el archivo SVG base '{svg_path}': {e}")
            continue

        # Lista para almacenar las bounding boxes de los elementos ya insertados
        inserted_bboxes = []

        # Procesar cada detección
        for x, y, w, h, rotation, class_name in detections:
            svg_class_path = os.path.join(library_path, "general", f"{class_name}.svg")
            if not os.path.exists(svg_class_path):
                print(f"SVG para la clase '{class_name}' no encontrado en {svg_class_path}.")
                continue

            # Verificar si esta bounding box ya contiene un elemento insertado
            overlapping = False
            for bx, by, bw, bh in inserted_bboxes:
                if not (x + w < bx or bx + bw < x or y + h < by or by + bh < y):  # Comprueba intersección
                    overlapping = True
                    break

            if overlapping:
                print(f"Bounding box ({x}, {y}, {w}, {h}) se solapa con un elemento existente. Se omite.")
                continue

            try:
                # Cargar el SVG de la clase detectada
                class_tree = ET.parse(svg_class_path)
                class_root = class_tree.getroot()

                # Limpiar namespaces del SVG de la clase
                limpiar_namespaces(class_root)

                # Obtener dimensiones del SVG de la clase
                svg_width = parse_dimension(class_root.attrib.get("width", "100px"))
                svg_height = parse_dimension(class_root.attrib.get("height", "100px"))

                # Calcular escalas manteniendo la relación de aspecto
                scale = min(w / svg_width, h / svg_height)

                # Crear un grupo (<g>) con las transformaciones necesarias
                cx, cy = x + (w / 2), y + (h / 2)
                transform = f"translate({cx},{cy}) scale({scale}) translate(-{svg_width / 2},-{svg_height / 2})"
                if rotation != 0:
                    transform += f" rotate({rotation})"

                grupo = ET.Element('g', attrib={'transform': transform})

                # Agregar los elementos del SVG de la clase al grupo
                for elem in list(class_root):
                    grupo.append(elem)

                # Insertar el grupo en el archivo SVG base
                root.append(grupo)

                # Registrar la bounding box como insertada
                inserted_bboxes.append((x, y, w, h))

            except Exception as e:
                print(f"Error al procesar el SVG de la clase '{class_name}': {e}")
                continue

        # Generar la ruta de salida
        svg_filename = os.path.basename(svg_path)
        output_svg_path = os.path.join(output_directory, svg_filename)

        # Guardar el archivo SVG actualizado
        try:
            tree.write(output_svg_path, encoding="utf-8", xml_declaration=True)
            print(f"Archivo SVG actualizado guardado en: {output_svg_path}")
        except Exception as e:
            print(f"Error al guardar el archivo SVG en '{output_svg_path}': {e}")





def main():
    # OCR arguments
    input_path, workspace, ocr_output_path, ocr_language, ocr_confidence_threshold, template_library, template_output_path, template_confidence_threshold, iou_confidence_threshold = ReSin_config()

    # OCR
    processed_images = text_detection(input_path, ocr_output_path, ocr_language, ocr_confidence_threshold)

    # Template matching
    TemplateMatching(processed_images, template_output_path, template_library, template_confidence_threshold, iou_confidence_threshold)

    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()