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
    arguments = parser.parse_args()

    s_input_path = arguments.input
    s_language = arguments.language.split(',')

    return s_input_path, s_language


def text_detection (input_path, language):
    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(input_path, input_image)

            # OCR.
            with Image.open(image_path) as image:
                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()

                detection = run_ocr([image], [language], det_model, det_processor, rec_model, rec_processor)

                # TODO: En el último paso de ReSin toda esta parte en adelante será modificada
                # SVG: Text write.
                output_name = os.path.splitext(input_image)[0] + ".svg"
                output_path = os.path.join('output/',output_name)

            image_width = detection[0].image_bbox[2] - detection[0].image_bbox[0]
            image_height = detection[0].image_bbox[3] - detection[0].image_bbox[1]
            
            output_svg = svg.Drawing(output_path, size=(image_width, image_height))

            for text_line in detection[0].text_lines:
                text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.5

                output_svg.add(output_svg.text(text_line.text,
                                               insert=(text_x, text_y),
                                               text_anchor = "middle",
                                               alignment_baseline = "middle",
                                               font_size = font_height))
            output_svg.save()
            

def main():
    # OCR arguments
    input_path, language = ReSin_config()

    # OCR
    text_detection(input_path, language)
    
    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()