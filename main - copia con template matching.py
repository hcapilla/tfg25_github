# Global
import argparse
import os
from collections import Counter

# SVG 
import svgwrite as svg

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
                output_images.append(output_path_PNG)

    print("OCR text detection completado.")
    return output_images

def calculate_iou(box1, box2):
    """
    Calcula la intersección sobre la unión (IoU) entre dos cajas delimitadoras.
    
    Args:
        box1 (tuple): Rectángulo 1 (x_min, y_min, width, height).
        box2 (tuple): Rectángulo 2 (x_min, y_min, width, height).
    
    Returns:
        float: Valor de IoU.
    """
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Coordenadas de la intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calcular área de la intersección
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Calcular área de la unión
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def TemplateMatching(processed_images, output_path, library_path):
    """
    Realiza Template Matching en las imágenes procesadas, permite al usuario clasificar la región seleccionada,
    realiza rotaciones y aplica IoU para filtrar resultados.

    Args:
        processed_images (list): Lista de rutas a las imágenes procesadas.
        output_path (str): Ruta de la carpeta de salida para guardar los resultados.
        library_path (str): Ruta de la carpeta de la librería de templates.
    """
    if not processed_images:
        print("No hay imágenes procesadas para analizar.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Cargar la primera imagen para seleccionar ROI
    first_image_path = processed_images[0]
    input_image = cv2.imread(first_image_path)
    if input_image is None:
        print(f"Error al cargar la imagen: {first_image_path}")
        return

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Seleccionar la ROI (Región de Interés)
    roi = cv2.selectROI("Selecciona la plantilla", input_image, showCrosshair=True)
    cv2.destroyWindow("Selecciona la plantilla")

    # Extraer la plantilla seleccionada
    template = gray_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    if template.size == 0:
        print("Error: No se seleccionó una plantilla válida.")
        return

    # Pedir al usuario la clase del elemento
    class_name = input("Introduce la clase a la que pertenece el elemento seleccionado: ")
    if not class_name:
        print("Clase no válida.")
        return

    # Crear carpeta para la clase si no existe en la librería
    class_folder = os.path.join(library_path, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Guardar el template en la librería
    template_path = os.path.join(class_folder, "template.png")
    cv2.imwrite(template_path, template)
    print(f"Template guardado en la clase '{class_name}'.")

    # Configuración de coincidencia
    match_method = cv2.TM_CCOEFF_NORMED
    threshold = 0.5
    iou_threshold = 0.5  # Umbral de IoU para filtrar duplicados

    # Procesar cada imagen en processed_images
    for image_path in processed_images:
        input_image = cv2.imread(image_path)
        if input_image is None:
            print(f"Advertencia: No se pudo cargar la imagen {image_path}. Se omitirá.")
            continue

        original_image = input_image.copy()
        detections = []

        # Realizar detección rotando el template
        for rotation in [0, 90, 180, 270]:
            rotated_template = template
            if rotation != 0:
                rotated_template = cv2.rotate(template, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])

            # Realizar la coincidencia de plantillas
            result = cv2.matchTemplate(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), rotated_template, match_method)

            # Encontrar ubicaciones que cumplan con el umbral
            locations = np.where(result >= threshold)

            for point in zip(*locations[::-1]):  # Invertir para obtener (x, y)
                top_left = point
                detections.append((top_left[0], top_left[1], rotated_template.shape[1], rotated_template.shape[0]))

        # Aplicar IoU para filtrar duplicados
        filtered_detections = []
        for i, det1 in enumerate(detections):
            keep = True
            for j, det2 in enumerate(filtered_detections):
                if calculate_iou(det1, det2) > iou_threshold:
                    keep = False
                    break
            if keep:
                filtered_detections.append(det1)

        # Dibujar rectángulos en las detecciones finales
        for x, y, w, h in filtered_detections:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Azul fuerte

        # Guardar la imagen con las coincidencias
        output_image_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{class_name}_detected.jpg"
        output_image_path = os.path.join(output_path, output_image_name)
        cv2.imwrite(output_image_path, original_image)

    print("Template Matching completado.")

def main():
    # OCR arguments
    input_path, workspace, ocr_output_path, ocr_language, ocr_confidence_threshold, template_library, template_output_path = ReSin_config()

    # OCR
    processed_images = text_detection(input_path, ocr_output_path, ocr_language, ocr_confidence_threshold)

    # Template matching
    TemplateMatching(processed_images, template_output_path, template_library)

    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()