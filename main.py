# Program
import argparse
import os
from collections import Counter

# SVG 
import svgwrite as svg

# OCR
from PIL import Image, ImageDraw, ImageFont
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# AMBER
from ultralytics import YOLO

def ReSin_config():
    parser = argparse.ArgumentParser(description="ReSin configuration")

    # OCR.
    parser.add_argument('--input', type=str, default='input/', help='Input image folder')
    parser.add_argument('--output', type=str, default='output/', help='Output image folder')
    parser.add_argument('--language', type=str, default='en', help='Languages')
    parser.add_argument('--ocr_confidence_threshold', type=float, default='0.7', help='OCR Confidence threshold')
    arguments = parser.parse_args()

    s_input_path = arguments.input
    s_output_path = arguments.output
    s_language = arguments.language.split(',')
    s_ocr_confidence_threshold = arguments.ocr_confidence_threshold

    return s_input_path, s_output_path, s_language, s_ocr_confidence_threshold

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
    
def deleteByThreshold(detection, ocr_confidence_threshold):
    detection[0].text_lines = [line for line in detection[0].text_lines if line.confidence >= ocr_confidence_threshold]

    return detection

def createOutputName(output_dir, base_name):
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

    return output_images

def model_apply_on_images(image_paths):

    model = YOLO("C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Amber Detection Model/Amber4/weights/best.pt")

    for image_path in image_paths:
        results = model.predict(source=image_path, conf=0.05)
        with Image.open(image_path) as original_image:
            draw = ImageDraw.Draw(original_image)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confs = result.boxes.conf.cpu().numpy()  # Confidences
                classes = result.boxes.cls.cpu().numpy()  # Class IDs
                names = model.names  # Obtener nombres de las clases

                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[int(cls)]} {conf:.2f}"
                    color = (0, 255, 0)  # Color verde para las cajas

                    # Dibujar la caja
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # Dibujar el texto
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)  # Cargar fuente
                    except IOError:
                        font = ImageFont.load_default()  # Usar fuente predeterminada si no se encuentra "arial.ttf"

                    # Calcular la posición del texto y el fondo
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    text_background = [(text_bbox[0], text_bbox[1] - 2), (text_bbox[2] + 2, text_bbox[3])]
                    draw.rectangle(text_background, fill=color)
                    draw.text((x1, y1), label, fill="black", font=font)

            # Guardar la imagen procesada con el sufijo "_Amber"
            output_path = os.path.splitext(image_path)[0] + "_Amber.png"
            original_image.save(output_path)
            print(f"Modelo aplicado a {image_path}, resultado guardado en {output_path}")

def model_apply_on_images_by_quadrants(image_paths):

    model = YOLO("C:/Users/Mi Pc/OneDrive - UAB/Documentos/TFG 25/Amber Detection Model/Amber4/weights/best.pt")

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            third_width, third_height = width // 3, height // 3

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

            # Procesar cada cuadrante con el modelo AMBER
            for quadrant, offset_x, offset_y, quadrant_id in quadrants:
                print(f"Processing quadrant: {quadrant_id}")
                results = model.predict(source=quadrant, conf=0.05)

                draw = ImageDraw.Draw(img)

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    names = model.names

                    for box, conf, cls in zip(boxes, confs, classes):
                        # Ajustar coordenadas con el desplazamiento del cuadrante
                        x1, y1, x2, y2 = [int(coord + offset_x if i % 2 == 0 else coord + offset_y)
                                          for i, coord in enumerate(box)]
                        label = f"{names[int(cls)]} {conf:.2f}"
                        color = (0, 255, 0)

                        # Dibujar la Bounding Box
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                        # Dibujar el texto
                        try:
                            font = ImageFont.truetype("arial.ttf", 16)
                        except IOError:
                            font = ImageFont.load_default()

                        text_size = font.getbbox(label)
                        text_background = [(x1, y1 - text_size[3] - 2), (x1 + text_size[2] + 2, y1)]
                        draw.rectangle(text_background, fill=color)
                        draw.text((x1, y1 - text_size[3]), label, fill="black", font=font)

            # Guardar la imagen procesada con el sufijo "_Amber"
            output_path = os.path.splitext(image_path)[0] + "_Amber.png"
            img.save(output_path)
            print(f"Modelo aplicado a {image_path}, resultado guardado en {output_path}")



def main():
    # OCR arguments
    input_path, output_path, language, ocr_confidence_threshold = ReSin_config()

    # OCR
    processed_images = text_detection(input_path, output_path, language, ocr_confidence_threshold)
    
    # Amber
    model_apply_on_images_by_quadrants(processed_images)

    print('Done')


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()