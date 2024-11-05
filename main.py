# Program
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# SVG 
import svgwrite as svg

# OCR
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import torch

# Forzar todos los tensores a float32
torch.set_default_dtype(torch.float32)

def ReSin_config():
    parser = argparse.ArgumentParser(description="ReSin configuration")

    # OCR.
    parser.add_argument('--input', type=str, default='input/', help='Image folder')
    parser.add_argument('--language', type=str, default='en,it', help='Languages')
    parser.add_argument('--ocr_confidence_threshold', type=float, default='0.72', help='OCR Confidence threshold')
    arguments = parser.parse_args()

    s_input_path = arguments.input
    s_language = arguments.language.split(',')
    s_ocr_confidence_threshold = arguments.ocr_confidence_threshold

    return s_input_path, s_language, s_ocr_confidence_threshold

def TEST_showConfidence(detection, quadrant_id):
    text_lines = detection[0].text_lines
    text_lines_sorted = sorted(text_lines, key=lambda line: line.confidence)

    for item in text_lines_sorted:
        # Determinar el color basado en el valor de confianza
        if item.confidence < 0.72:
            color = "\033[91m"  # Rojo
        elif item.confidence < 0.8:
            color = "\033[93m"  # Amarillo
        else:
            color = "\033[94m"  # Azul

        # Resetear color al final del número
        reset_color = "\033[0m"

        # Imprimir el texto con el valor de confianza coloreado
        print(f"Text: {item.text}, Confidence: {color}{item.confidence:.2f}{reset_color}, Quadrant: {quadrant_id}")

def TEST_drawBoundingBoxes(text_line, output_svg):
    # NOTE: those are common thresholds. May vary depending on the number of quadrants and the image
    if text_line['confidence'] < 0.7:
        boundingBox_color = "red" 
    elif text_line['confidence'] < 0.82: 
        boundingBox_color = "yellow"
    else:
        boundingBox_color = "blue"
    
    output_svg.add(output_svg.polygon(
        points=text_line['polygon'],
        fill="none",
        stroke=boundingBox_color,
        stroke_width=2
    ))
    

def deleteByThreshold(detection, ocr_confidence_threshold):
    detection[0].text_lines = [line for line in detection[0].text_lines if line.confidence >= ocr_confidence_threshold]

    return detection

def createOutputName(output_dir, base_name):
    output_name = f"{base_name}.svg"
    output_path = os.path.join(output_dir, output_name)
    
    if not os.path.exists(output_path):
        return output_path

    counter = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}({counter}).svg")):
        counter += 1
    
    return os.path.join(output_dir, f"{base_name}({counter}).svg")

def calculate_central_weight(bbox, quadrant_center):
    text_center_x = (bbox[0] + bbox[2]) / 2
    text_center_y = (bbox[1] + bbox[3]) / 2
    distance_x = abs(text_center_x - quadrant_center[0])
    distance_y = abs(text_center_y - quadrant_center[1])
    max_distance = max(quadrant_center)
    central_weight = 1 - ((distance_x + distance_y) / (2 * max_distance))
    return max(0, min(central_weight, 1))

def process_quadrant(quadrant, offset_x, offset_y, quadrant_center, language, ocr_confidence_threshold, quadrant_id):
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    # Convertir modelos y datos a float32 explícitamente
    det_model = det_model.to(torch.float32)
    rec_model = rec_model.to(torch.float32)

    quadrant = quadrant.convert("RGB")  # Asegurarse de que esté en RGB
    detection = run_ocr([quadrant], [language], det_model, det_processor, rec_model, rec_processor)
    detection = deleteByThreshold(detection, ocr_confidence_threshold)

    adjusted_text_lines = []
    for text_line in detection[0].text_lines:
        central_weight = calculate_central_weight(text_line.bbox, quadrant_center)
        adjusted_text_line = {
            "text": text_line.text,
            "confidence": text_line.confidence * central_weight,
            "bbox": [coord + offset_x if j % 2 == 0 else coord + offset_y 
                     for j, coord in enumerate(text_line.bbox)],
            "polygon": [(x + offset_x, y + offset_y) for x, y in text_line.polygon]
        }
        adjusted_text_lines.append(adjusted_text_line)

    TEST_showConfidence(detection, quadrant_id)
    
    return adjusted_text_lines

def image_preprocessing(image_path, language, ocr_confidence_threshold):
    with Image.open(image_path) as img:
        width, height = img.size
        third_width, third_height = width // 3, height // 3

        # Definimos los 9 cuadrantes, incluyendo un identificador de cuadrante (quadrant_id)
        quadrants = [
            (img.crop((0, 0, third_width, third_height)), 0, 0, "Top-left"),
            (img.crop((third_width, 0, 2 * third_width, third_height)), third_width, 0, "Top-center"),
            (img.crop((2 * third_width, 0, width, third_height)), 2 * third_width, 0, "Top-right"),
            (img.crop((0, third_height, third_width, 2 * third_height)), 0, third_height, "Center-left"),
            (img.crop((third_width, third_height, 2 * third_width, 2 * third_height)),
             third_width, third_height, "Center-center"),
            (img.crop((2 * third_width, third_height, width, 2 * third_height)), 2 * third_width, third_height, "Center-right"),
            (img.crop((0, 2 * third_height, third_width, height)), 0, 2 * third_height, "Bottom-left"),
            (img.crop((third_width, 2 * third_height, 2 * third_width, height)),
             third_width, 2 * third_height, "Bottom-center"),
            (img.crop((2 * third_width, 2 * third_height, width, height)), 2 * third_width, 2 * third_height, "Bottom-right")
        ]

        full_detections = []
        
        # Procesar OCR en paralelo para cada cuadrante y pasar quadrant_id
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_quadrant,
                    quadrant,
                    offset_x,
                    offset_y,
                    ((offset_x + third_width // 2), (offset_y + third_height // 2)),
                    language,
                    ocr_confidence_threshold,
                    quadrant_id
                )
                for quadrant, offset_x, offset_y, quadrant_id in quadrants
            ]

            for future in as_completed(futures):
                full_detections.extend(future.result())

        final_detections = filter_detections(full_detections)

        return final_detections, width, height


def filter_detections(detections, overlap_threshold=10):
    unique_detections = []
    for detection in detections:
        keep = True
        for unique in unique_detections:
            if boxes_overlap(detection["bbox"], unique["bbox"], overlap_threshold):
                if detection["confidence"] > unique["confidence"]:
                    unique_detections.remove(unique)
                    unique_detections.append(detection)
                keep = False
                break
        if keep:
            unique_detections.append(detection)
    return unique_detections

def boxes_overlap(bbox1, bbox2, threshold):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    horizontal_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    vertical_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return horizontal_overlap > threshold and vertical_overlap > threshold


def text_detection(input_path, language, ocr_confidence_threshold):
    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(input_path, input_image)
            
            detections, image_width, image_height = image_preprocessing(image_path, language, ocr_confidence_threshold)
            
            # Final SVG generation
            output_path = createOutputName('output', os.path.splitext(input_image)[0])
            output_svg = svg.Drawing(output_path, size=(image_width, image_height))
            
            for text_line in detections:
                text_x = (text_line["bbox"][0] + text_line["bbox"][2]) / 2
                text_y = (text_line["bbox"][1] + text_line["bbox"][3]) / 2
                font_height = (text_line["bbox"][3] - text_line["bbox"][1]) * 0.9
                
                #TEST_drawBoundingBoxes(text_line, output_svg)

                output_svg.add(output_svg.text(
                    text_line["text"],
                    insert=(text_x, text_y),
                    text_anchor="middle",
                    alignment_baseline="middle",
                    font_size="15px",
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