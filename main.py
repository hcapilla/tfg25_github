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
    text_lines = detection[0].text_lines
    text_lines_sorted = sorted(text_lines, key=lambda line: line.confidence)

    for item in text_lines_sorted:
        print(f"Text: {item.text}, Confidence: {item.confidence}")

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

def process_quadrant(quadrant, offset_x, offset_y, language, ocr_confidence_threshold):

    # OCR
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    detection = run_ocr([quadrant], [language], det_model, det_processor, rec_model, rec_processor)
    detection = deleteByThreshold(detection, ocr_confidence_threshold)

    adjusted_text_lines = []
    for text_line in detection[0].text_lines:
        adjusted_text_line = {
            "text": text_line.text,
            "confidence": text_line.confidence,
            "bbox": [coord + offset_x if j % 2 == 0 else coord + offset_y 
                     for j, coord in enumerate(text_line.bbox)],
            "polygon": [(x + offset_x, y + offset_y) for x, y in text_line.polygon]
        }
        adjusted_text_lines.append(adjusted_text_line)
    
    return adjusted_text_lines

def image_preprocessing(image_path, language, ocr_confidence_threshold):
    with Image.open(image_path) as img:
        width, height = img.size
        quadrant_width = width // 2
        quadrant_height = height // 2

        quadrants = [
            (img.crop((0, 0, quadrant_width, quadrant_height)), 0, 0),                                      # Top-left
            (img.crop((quadrant_width, 0, width, quadrant_height)), quadrant_width, 0),                     # Top-right
            (img.crop((0, quadrant_height, quadrant_width, height)), 0, quadrant_height),                   # Bottom-left
            (img.crop((quadrant_width, quadrant_height, width, height)), quadrant_width, quadrant_height)   # Bottom-right
        ]

        full_detections = []
        
        # Paralel OCR processing
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_quadrant, quadrant, offset_x, offset_y, language, ocr_confidence_threshold) 
                       for quadrant, offset_x, offset_y in quadrants]

            for future in as_completed(futures):
                full_detections.extend(future.result())

        return full_detections, width, height


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
                
                TEST_drawBoundingBoxes(text_line, output_svg)

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