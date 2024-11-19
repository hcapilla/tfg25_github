# Program
import argparse
import os

# Multi-thread
from concurrent.futures import ThreadPoolExecutor

# Calculus
from math import sqrt

# SVG 
import svgwrite as svg

# OCR
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

def ReSin_config():
    parser = argparse.ArgumentParser(description="ReSin configuration")

    # OCR.
    parser.add_argument('--input', type=str, default='input/', help='Image folder')
    parser.add_argument('--language', type=str, default='en', help='Languages')
    parser.add_argument('--ocr_confidence_threshold', type=float, default='0.72', help='OCR Confidence threshold')
    arguments = parser.parse_args()

    s_input_path = arguments.input
    s_language = arguments.language.split(',')
    s_ocr_confidence_threshold = arguments.ocr_confidence_threshold

    return s_input_path, s_language, s_ocr_confidence_threshold

def TEST_showConfidence(detection):
    # Acceder a los text_lines de detection[0]
    text_lines = detection[0].text_lines

    # Ordenar text_lines por el atributo 'confidence' de menor a mayor
    text_lines_ordenado = sorted(text_lines, key=lambda line: line.confidence)

    # Mostrar la lista ordenada
    for item in text_lines_ordenado:
        print(f"Text: {item.text}, Confidence: {item.confidence}")

def TEST_drawBoundingBoxes(text_line, output_svg):
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
    
def deleteByThreshold(text_lines, ocr_confidence_threshold):
    # Filtra las líneas de texto basándose en el umbral de confianza
    return [line for line in text_lines if line.confidence >= ocr_confidence_threshold]

def createOutputName(output_dir, base_name):
    output_name = f"{base_name}.svg"
    output_path = os.path.join(output_dir, output_name)
    
    # Si no existe un archivo con este nombre, lo retornamos
    if not os.path.exists(output_path):
        return output_path

    # Si ya existe, encontrar el número más pequeño disponible
    counter = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}({counter}).svg")):
        counter += 1
    
    return os.path.join(output_dir, f"{base_name}({counter}).svg")

def createQuadrants(image, nQuadrantsH, nQuadrantsV):
    image_height = image.height
    image_width = image.width
    sizeQuadrantsH = image_width // nQuadrantsH
    sizeQuadrantsV = image_height // nQuadrantsV

    quadrants = []

    # o1
    for qV in range(nQuadrantsV):
        for qH in range(nQuadrantsH):
            x1 = qH * sizeQuadrantsH
            x2 = x1 + sizeQuadrantsH
            y1 = qV * sizeQuadrantsV
            y2 = y1 + sizeQuadrantsV

            crop = image.crop((x1, y1, x2, y2))
            quadrants.append({'Type': 'o1', 'Coordinates': (x1, y1, x2, y2), 'Image': crop})

    # o2
    # o2 - H
    o2_horizontals = []
    if nQuadrantsH >= 2:
        for qV in range(nQuadrantsV):
            for qH in range(nQuadrantsH - 1):
                x1 = (qH + 0.5) * sizeQuadrantsH
                x2 = x1 + sizeQuadrantsH
                y1 = qV * sizeQuadrantsV
                y2 = y1 + sizeQuadrantsV

                crop = image.crop((x1, y1, x2, y2))
                quadrant = {'Type': 'o2-H', 'Coordinates': (x1, y1, x2, y2), 'Image': crop}
                quadrants.append(quadrant)
                o2_horizontals.append(quadrant)

    # o2 - V
    o2_verticals = []
    if nQuadrantsV >= 2:
        for qV in range(nQuadrantsV - 1):
            for qH in range(nQuadrantsH):
                x1 = qH * sizeQuadrantsH
                x2 = x1 + sizeQuadrantsH
                y1 = (qV + 0.5) * sizeQuadrantsV
                y2 = y1 + sizeQuadrantsV

                crop = image.crop((x1, y1, x2, y2))
                quadrant = {'Type': 'o2-V', 'Coordinates': (x1, y1, x2, y2), 'Image': crop}
                quadrants.append(quadrant)
                o2_verticals.append(quadrant)

    # o3
    # o3 - H
    for h1, h2 in zip(o2_horizontals[:-1], o2_horizontals[1:]):
        if h1['Coordinates'][1] == h2['Coordinates'][1]:
            x1 = (h1['Coordinates'][2] + h2['Coordinates'][0]) / 2 - sizeQuadrantsH / 2
            x2 = x1 + sizeQuadrantsH
            y1 = h1['Coordinates'][1]
            y2 = y1 + sizeQuadrantsV

            crop = image.crop((x1, y1, x2, y2))
            quadrants.append({'Type': 'o3-H', 'Coordinates': (x1, y1, x2, y2), 'Image': crop})

    # o3 - V
    for v1, v2 in zip(o2_verticals[:-1], o2_verticals[1:]):
        if v1['Coordinates'][0] == v2['Coordinates'][0]:
            x1 = v1['Coordinates'][0]
            x2 = x1 + sizeQuadrantsH
            y1 = (v1['Coordinates'][3] + v2['Coordinates'][1]) / 2 - sizeQuadrantsV / 2
            y2 = y1 + sizeQuadrantsV

            crop = image.crop((x1, y1, x2, y2))
            quadrants.append({'Type': 'o3-V', 'Coordinates': (x1, y1, x2, y2), 'Image': crop})

    return quadrants

def calculate_center(coordinates):
    x1, y1, x2, y2 = coordinates
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def merge_text_lines(quadrants, text_lines_with_quadrant, radius=10):
    filtered_lines = []

    for line, quadrant_id in text_lines_with_quadrant:
        keep = True
        for other_line, other_quadrant_id in text_lines_with_quadrant:
            if line == other_line:
                continue

            dist = distance((line.bbox[0], line.bbox[1]), (other_line.bbox[0], other_line.bbox[1]))
            if dist <= radius:
                center_line = distance((line.bbox[0], line.bbox[1]), calculate_center(quadrants[quadrant_id]["Coordinates"]))
                center_other = distance((other_line.bbox[0], other_line.bbox[1]), calculate_center(quadrants[other_quadrant_id]["Coordinates"]))

                if center_line > center_other:
                    keep = False
                    break

        if keep:
            filtered_lines.append(line)

    return filtered_lines

def text_detection(input_path, language, ocr_confidence_threshold):
    # Cargar modelos una sola vez
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg', 'jpeg')):
            image_path = os.path.join(input_path, input_image)

            with Image.open(image_path) as image:
                nQuadrantsH, nQuadrantsV = 1, 3
                quadrants = createQuadrants(image, nQuadrantsH, nQuadrantsV)

                # OCR en cada cuadrante
                text_lines_with_quadrant = []
                for quadrant_id, quadrant in enumerate(quadrants):
                    det = run_ocr(
                        [image],
                        [language],
                        det_model,
                        det_processor,
                        rec_model,
                        rec_processor
                    )[0]
                    #det.text_lines = deleteByThreshold(det.text_lines, ocr_confidence_threshold)
                    
                    # Asociar cada línea con el cuadrante correspondiente
                    for line in det.text_lines:
                        text_lines_with_quadrant.append((line, quadrant_id))

                # Consolidar texto final
                final_text_lines = merge_text_lines(quadrants, text_lines_with_quadrant)

                # Procesar texto final para generar la salida
                output_path = createOutputName("output", os.path.splitext(input_image)[0])
                image_width = image.width
                image_height = image.height
                output_svg = svg.Drawing(output_path, size=(image_width, image_height))

                for text_line in final_text_lines:
                    text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                    text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                    font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.9

                    TEST_drawBoundingBoxes(text_line, output_svg)

                    output_svg.add(output_svg.text(
                        text_line.text,
                        insert=(text_x, text_y),
                        text_anchor="middle",
                        alignment_baseline="middle",
                        font_size=font_height,
                        font_weight="bold",
                        font_family="Tahoma"
                    ))

                output_svg.save()
            

def main():
    # OCR arguments
    input_path, language, ocr_confidence_threshold = ReSin_config()

    # OCR
    text_detection(input_path, language, ocr_confidence_threshold)
    
    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()