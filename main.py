# Program
import argparse
import os

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
    parser.add_argument('--ocr_confidence_threshold', type=float, default='0.7', help='OCR Confidence threshold')
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
    
def deleteByThreshold(detection, ocr_confidence_threshold):
    detection[0].text_lines = [line for line in detection[0].text_lines if line.confidence >= ocr_confidence_threshold]

    return detection


def text_detection (input_path, language, ocr_confidence_threshold):
    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(input_path, input_image)

            # OCR.
            with Image.open(image_path) as image:
                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()

                detection = run_ocr([image], [language], det_model, det_processor, rec_model, rec_processor)
                detection = deleteByThreshold(detection, ocr_confidence_threshold)

                # TODO: En el último paso de ReSin toda esta parte en adelante será modificada
                # SVG: Text write.
                # [MEJORA] Image_preprocessing: Dividir la imagen en 4 cuadrantes y analizarlas por separado
                output_name = os.path.splitext(input_image)[0] + ".svg"
                output_path = os.path.join('output/',output_name)

            #TEST_showConfidence(detection)

            image_width = detection[0].image_bbox[2] - detection[0].image_bbox[0]
            image_height = detection[0].image_bbox[3] - detection[0].image_bbox[1]
            
            output_svg = svg.Drawing(output_path, size=(image_width, image_height))

            for text_line in detection[0].text_lines:
                text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.9

                TEST_drawBoundingBoxes(text_line, output_svg)

                output_svg.add(output_svg.text(text_line.text,
                                               insert=(text_x, text_y),
                                               text_anchor = "middle",
                                               alignment_baseline = "middle",
                                               font_size = font_height,
                                               font_weight = "bold",
                                               font_family = "Tahoma"))
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