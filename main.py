# Global
import argparse
import os
from collections import Counter
import random

# SVG 
import svgwrite as svg 
from xml.etree import ElementTree as ET
import copy

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
    """
    Configures the ReSin application by parsing arguments, initializing workspace paths,
    and returning the required settings.

    Returns:
        tuple: Paths and configurations for the application.
    """
    def initializeWorkspacePaths(workspace, input_path, ocr_output_path, template_library, template_output_path, connections_output_path):
        input_path = os.path.join(workspace, input_path)
        ocr_output_path = os.path.join(workspace, ocr_output_path)
        template_library = os.path.join(workspace, template_library)
        template_output_path = os.path.join(workspace, template_output_path)
        connections_output_path = os.path.join(workspace, connections_output_path)

        for path in [input_path, ocr_output_path, template_library, template_output_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        return input_path, ocr_output_path, template_library, template_output_path, connections_output_path

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

    # PATH CREATION
    parser.add_argument('--connections_output', type=str, default='output/', help='Output Post-Connections folder - Finished process')

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

    s_connections_output_path = arguments.connections_output

    s_input_path, s_ocr_output_path, s_template_library, s_template_output_path, s_connections_output_path = initializeWorkspacePaths(
        s_workspace_path, s_input_path, s_ocr_output_path, s_template_library, s_template_output_path, s_connections_output_path
    )

    return s_input_path, s_workspace_path, s_ocr_output_path, s_ocr_language, s_ocr_confidence_threshold, s_template_library, s_template_output_path, s_template_confidence_threshold, s_iou_confidence_threshold, s_connections_output_path

def print_amber(text):
    amber_rgb = (255, 191, 0)
    print(f"\033[38;2;{amber_rgb[0]};{amber_rgb[1]};{amber_rgb[2]}m{text}\033[0m")

def get_dominant_color(image_path):
        """
        Calculates the dominant color of an image. If the most frequent color is black, the second most frequent color is returned.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Dominant color in (R, G, B) format.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be read. Check the file path.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)
        pixel_counts = Counter(map(tuple, pixels))
        sorted_colors = pixel_counts.most_common()

        for color, _ in sorted_colors:
            if color != (0, 0, 0):
                return color

        return (0, 0, 0)

# OCR
def text_detection(input_path, output_path, language, ocr_confidence_threshold):
    """
    Detects text in images within the input directory, performs OCR, and generates corresponding SVG and PNG outputs.

    Args:
        input_path (str): Path to the directory containing input images.
        output_path (str): Path to the directory where the outputs will be saved.
        language (str): Language for the OCR model.
        ocr_confidence_threshold (float): Minimum confidence threshold for text detection.

    Returns:
        list: A list of outputs, each containing the paths to the PNG, SVG, and text detections.
    """

    def deleteByThreshold(detection, ocr_confidence_threshold):
        """
        Filters text lines from the OCR detection based on confidence threshold and specific text conditions.

        Args:
            detection (list): OCR detection results containing text lines.
            ocr_confidence_threshold (float): Minimum confidence threshold for retaining a text line.

        Returns:
            list: Updated detection results with filtered text lines.
        """
        def is_valid_line(line):
            if line.confidence < ocr_confidence_threshold:
                return False
            if len(line.text) == 1:
                return False
            if len(line.text) == 2:
                return False
            if len(set(line.text)) == 1:
                return False
            if "(" in line.text and ")" not in line.text:
                line.text = line.text.split("(")[0]
            if "[" in line.text and "]" not in line.text:
                line.text = line.text.split("[")[0]
            return True

        detection[0].text_lines = [line for line in detection[0].text_lines if is_valid_line(line)]
        return detection

    def createOutputName(output_dir, base_name):
        """
        Generates unique output file names for SVG and PNG formats in the specified directory.

        Args:
            output_dir (str): Directory where the files will be saved.
            base_name (str): Base name for the output files.

        Returns:
            tuple: Paths to the SVG and PNG files with unique names.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_name_svg = f"{base_name}.svg"
        output_name_png = f"{base_name}.png"

        output_path_svg = os.path.join(output_dir, output_name_svg)
        output_path_png = os.path.join(output_dir, output_name_png)

        if not os.path.exists(output_path_svg) and not os.path.exists(output_path_png):
            return output_path_svg, output_path_png

        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{counter}.svg")) or os.path.exists(os.path.join(output_dir, f"{base_name}_{counter}.png")):
            counter += 1

        output_path_svg = os.path.join(output_dir, f"{base_name}_{counter}.svg")
        output_path_png = os.path.join(output_dir, f"{base_name}_{counter}.png")

        return output_path_svg, output_path_png
    
    output_images = []

    for input_image in os.listdir(input_path):
        if input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_path, input_image)

            with Image.open(image_path) as image:
                output_path_SVG, output_path_PNG = createOutputName(output_path, os.path.splitext(input_image)[0])

                most_color = get_dominant_color(image_path)

                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()

                detection = run_ocr([image], [language], det_model, det_processor, rec_model, rec_processor)
                detection = deleteByThreshold(detection, ocr_confidence_threshold)

                image_width = image._size[0]
                image_height = image._size[1]

                output_svg = svg.Drawing(output_path_SVG, size=(image_width, image_height), profile='full')
                output_svg['xmlns'] = 'http://www.w3.org/2000/svg'

                most_color_hex = '#{:02x}{:02x}{:02x}'.format(*most_color)

                background_layer = output_svg.g(id="background-layer")
                text_layer = output_svg.g(id="text-layer")

                output_svg.add(background_layer)
                output_svg.add(text_layer)

                background_layer.add(output_svg.rect(
                    insert=(0, 0),
                    size=(image_width, image_height),
                    fill=most_color_hex
                ))

                text_detections = []

                for text_line in detection[0].text_lines:
                    x_min, y_min = int(text_line.bbox[0]), int(text_line.bbox[1])
                    x_max, y_max = int(text_line.bbox[2]), int(text_line.bbox[3])

                    for x in range(x_min, x_max):
                        for y in range(y_min, y_max):
                            image.putpixel((x, y), most_color)  # Use RGB tuple

                    text_x = (text_line.bbox[0] + text_line.bbox[2]) / 2
                    text_y = (text_line.bbox[1] + text_line.bbox[3]) / 2
                    font_height = (text_line.bbox[3] - text_line.bbox[1]) * 0.9

                    if 0 <= font_height <= 20:
                        font_size = 15
                    elif 20 < font_height <= 30:
                        font_size = 25
                    elif 30 < font_height <= 40:
                        font_size = 35
                    else:
                        font_size = int(font_height)

                    text_layer.add(output_svg.text(
                        text_line.text,
                        insert=(text_x, text_y),
                        text_anchor="middle",
                        alignment_baseline="middle",
                        font_size=font_size,
                        font_weight="bold",
                        font_family="Tahoma"
                    ))

                    text_detections.append({
                        "bbox": [x_min, y_min, x_max, y_max],
                        "text": text_line.text
                    })

                image.save(output_path_PNG)
                output_svg.save()

                output_images.append([output_path_PNG, output_path_SVG, text_detections])

    print_amber("OCR text detection done!")
    return output_images


# Template matching
def template_matching(processed_images, output_path, library_path, template_threshold, iou_threshold):
    """
    Performs template matching on processed images using a library of templates and handles user interaction
    for template addition and annotation.

    Args:
        processed_images (list): List of processed images with paths and detections.
        output_path (str): Directory where the outputs will be saved.
        library_path (str): Directory containing template libraries for each class.
        template_threshold (float): Minimum similarity threshold for template matching.
        iou_threshold (float): Minimum IoU threshold for filtering overlapping detections.

    Returns:
        list: List of outputs, each containing updated paths and detection results.
    """
    def calculate_iou(box1, box2, image, predominant_color):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes. If the boxes overlap,
        prioritizes the one with fewer pixels of the predominant color in the intersection region.

        Args:
            box1 (tuple): Tuple (x, y, width, height) of the first bounding box.
            box2 (tuple): Tuple (x, y, width, height) of the second bounding box.
            image (numpy.ndarray): Image array for pixel analysis.
            predominant_color (tuple): Tuple (B, G, R) representing the predominant color.

        Returns:
            float or int: IoU between the two boxes. If prioritizing a box, returns the index of the box
            with more pixels of the predominant color.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            return 0

        intersection_region = image[yi1:yi2, xi1:xi2]

        predominant_color_bgr = np.array(predominant_color, dtype=np.uint8)
        mask = np.all(intersection_region == predominant_color_bgr, axis=-1)
        predominant_pixels_count = np.sum(mask)

        region1 = image[y1:y1 + h1, x1:x1 + w1]
        region2 = image[y2:y2 + h2, x2:x2 + w2]

        mask1 = np.all(region1 == predominant_color_bgr, axis=-1)
        mask2 = np.all(region2 == predominant_color_bgr, axis=-1)

        count1 = np.sum(mask1)
        count2 = np.sum(mask2)

        if count1 > count2:
            return 1
        elif count2 > count1:
            return 2

        area1 = w1 * h1
        area2 = w2 * h2

        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def clean_namespaces(element):
        """
        Removes namespaces from XML elements.

        Args:
            element (xml.etree.ElementTree.Element): Root element of the XML tree.

        Returns:
            None: The function modifies the input element in place.
        """
        for elem in element.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        return element

    def parse_dimension(value):
        """
        Converts dimensions like '5.3646in', '120px', etc., into a float value in pixels.

        Args:
            value (str): A string representing a dimension with a unit (e.g., '5in', '10cm', '15px').

        Returns:
            float: The dimension converted into pixels.
        """
        if "in" in value:
            return float(value.replace("in", "")) * 96
        elif "cm" in value:
            return float(value.replace("cm", "")) * 37.7952755906
        elif "mm" in value:
            return float(value.replace("mm", "")) * 3.77952755906
        elif "px" in value:
            return float(value.replace("px", ""))
        else:
            return float(value)

    def insert_detected_svgs(detections, library_path, processed_images, output_directory, scale_factor=0.8):
        """
        Inserts detected SVG elements into corresponding base SVG files.

        If an SVG file for a detected class is not found, it draws the bounding box instead and prints a
        single message for missing files.
        Inserts elements into a separate "element_detections" layer, positioned between the background
        and text layers.

        Args:
            detections (list): List of detections in the format [(x, y, w, h, rotation, class_name, element_id), ...].
            library_path (str): Path to the folder containing SVG files for detected classes.
            processed_images (list): List of processed images with paths [(png_path, svg_path, res), ...].
            output_directory (str): Directory to save updated SVG files.
            scale_factor (float): Scaling factor to reduce the size of elements (default: 0.8).

        Returns:
            None
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)   

        missing_svgs = set()
        output_svg_paths = []   

        for processed_image in processed_images:
            if len(processed_image) < 2:
                print(f"Warning: Incorrect structure in processed_images: {processed_image}.")
                continue    

            png_path, svg_path = processed_image[:2]    

            if not os.path.exists(svg_path):
                print(f"The base SVG file '{svg_path}' does not exist. Skipping.")
                continue    

            try:
                # Parse base SVG
                tree = ET.parse(svg_path)
                root = tree.getroot()
                root = clean_namespaces(root)   

                # Find or create detection layer
                element_detections_layer = root.find(".//*[@id='element_detections']")
                if element_detections_layer is None:
                    element_detections_layer = ET.Element('g', {'id': 'element_detections'})
                    background_layer = root.find(".//*[@id='background-layer']")
                    text_layer = root.find(".//*[@id='text-layer']")

                    insert_pos = 0
                    if text_layer is not None:
                        insert_pos = list(root).index(text_layer)
                    elif background_layer is not None:
                        insert_pos = list(root).index(background_layer) + 1

                    root.insert(insert_pos, element_detections_layer)   

                # Process detections
                for detection in detections:
                    x, y, w, h, rotation, class_name, element_id = detection
                    svg_class_path = os.path.join(library_path, "general", f"{class_name}.svg") 

                    if not os.path.exists(svg_class_path):
                        missing_svgs.add(svg_class_path)
                        # Draw red bounding box
                        bbox_rect = ET.Element('rect', {
                            'x': str(x),
                            'y': str(y),
                            'width': str(w),
                            'height': str(h),
                            'fill': 'none',
                            'stroke': 'red',
                            'stroke-width': '2'
                        })
                        element_detections_layer.append(bbox_rect)
                        continue    

                    try:
                        # Parse template SVG
                        class_tree = ET.parse(svg_class_path)
                        class_root = class_tree.getroot()
                        class_root = clean_namespaces(class_root)   

                        # Get dimensions
                        if 'viewBox' in class_root.attrib:
                            _, _, vb_width, vb_height = map(float, class_root.attrib['viewBox'].split())
                        else:
                            vb_width = parse_dimension(class_root.get('width', '100'))
                            vb_height = parse_dimension(class_root.get('height', '100'))    

                        # Calculate scaling factors
                        scale_x = w / vb_width
                        scale_y = h / vb_height 

                        # Create transformation
                        transform = f"translate({x},{y}) scale({scale_x} {scale_y})"

                        # Add rotation around bounding box center if needed
                        if rotation != 0:
                            center_x = x + w/2
                            center_y = y + h/2
                            transform += f" rotate({rotation},{center_x},{center_y})"   

                        # Create group element
                        grupo = ET.Element('g', {
                            'transform': transform,
                            'id': f"{class_name}_{element_id}"
                        })  

                        # Copy all elements from template
                        for elem in class_root:
                            grupo.append(copy.deepcopy(elem))    

                        element_detections_layer.append(grupo)  

                    except Exception as e:
                        print(f"Error processing {class_name} (ID {element_id}): {str(e)}")
                        continue    

                # Save modified SVG
                output_filename = os.path.basename(svg_path)
                output_path = os.path.join(output_directory, output_filename)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)

            except Exception as e:
                print(f"Error processing {svg_path}: {str(e)}")
                continue    

        if missing_svgs:
            print("Missing SVG templates:")
            for missing in missing_svgs:
                print(f" - {missing}")  

        return output_path

    class_colors = {}
    output_images = []
    element_id = 1

    for image_pair in processed_images:
        if not isinstance(image_pair, list) or len(image_pair) < 3:
            print(f"Warning: Incorrect structure in processed_images: {image_pair}.")
            continue

        png_path, svg_path, texts = image_pair

        if not os.path.exists(png_path):
            print(f"Warning: PNG image {png_path} does not exist.")
            continue

        input_image = cv2.imread(png_path)
        if input_image is None:
            print(f"Warning: Could not load image {png_path}. It will be skipped.")
            continue

        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        most_color = get_dominant_color(png_path)
        most_color = tuple(map(int, most_color[::-1]))

        detections = []
        for class_name in os.listdir(library_path):
            class_folder = os.path.join(library_path, class_name)
            templates_folder = os.path.join(class_folder, "templates")

            if not os.path.exists(templates_folder):
                continue

            print(f"Accessing templates folder: {templates_folder}")

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
                            detections.append((bbox[0], bbox[1], bbox[2], bbox[3], rotation_angle, class_name, element_id))
                            element_id += 1

        all_detections = detections.copy()

        while True:
            temp_image = input_image.copy()
            for x, y, w, h, rotation, class_name, det_id in all_detections:
                if class_name not in class_colors:
                    class_colors[class_name] = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

                color = tuple(int(c) for c in class_colors[class_name].strip("rgb()").split(","))
                rotation_display = rotation if rotation in [90, 180, 270] else 0
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(temp_image, f"{class_name} ({rotation_display}) ID: {det_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

            roi = cv2.selectROI("Select the template", temp_image, showCrosshair=True)
            cv2.destroyWindow("Select the template")

            if roi[2] == 0 or roi[3] == 0:
                print("Selection finished or invalid region.")
                break

            template = gray_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            if template.size == 0:
                print("Error: No valid template selected. Please try again.")
                continue

            class_name = input("Enter the class of the selected element: ")
            if not class_name:
                print("Invalid class. Please try again.")
                continue

            class_folder = os.path.join(library_path, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            templates_folder = os.path.join(class_folder, "templates")
            if not os.path.exists(templates_folder):
                os.makedirs(templates_folder)

            cv2.imwrite(os.path.join(templates_folder, "template.png"), template)

            iteration = 0
            height, width = template.shape[:2]

            while width < 2 * template.shape[1] and height < 2 * template.shape[0]:
                resized_template = cv2.resize(template, (width, height))
                cv2.imwrite(os.path.join(templates_folder, f"template_increase_{iteration}.png"), resized_template)
                width += 2
                height += 2
                iteration += 1

            width, height = template.shape[1], template.shape[0]
            iteration = 0

            while width > template.shape[1] // 2 and height > template.shape[0] // 2:
                resized_template = cv2.resize(template, (max(width, 1), max(height, 1)))
                cv2.imwrite(os.path.join(templates_folder, f"template_decrease_{iteration}.png"), resized_template)
                width -= 2
                height -= 2
                iteration += 1

            print(f"Template variations saved in the 'templates' folder for class '{class_name}'.")

            print(f"Running template matching for the new class '{class_name}'...")
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
                        for existing_bbox in all_detections:
                            iou_or_priority = calculate_iou(bbox, existing_bbox[:4], input_image, most_color)

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
                            all_detections.append((bbox[0], bbox[1], bbox[2], bbox[3], rotation_angle, class_name, element_id))
                            element_id += 1

            print(f"Detections added for class '{class_name}'.")

        for x, y, w, h, rotation, class_name, det_id in all_detections:
            cv2.rectangle(input_image, (x, y), (x + w, y + h), most_color, -1)

        output_template_png_path = os.path.join(output_path, os.path.basename(png_path))
        cv2.imwrite(output_template_png_path, input_image)
        print(f"Image updated with the new detections: {output_template_png_path}")

        output_template_svg_path = insert_detected_svgs(all_detections, library_path, processed_images, output_path)

        output_images.append([output_template_png_path, output_template_svg_path, all_detections])

        print_amber("Template matching done!")

    return output_images


# Connections creation
def find_connections(processed_images, output_path):
    """
    Finds connections and terminals in processed images based on bounding box detections.

    Args:
        processed_images (list): List of processed image data with paths and detections.
        output_path (str): Directory to save processed connection data or results.

    Returns:
        list: A list of detected connections for each image.
    """
    def precompute_grid(detections, image_shape):
        """
        Crea una cuadrícula que indica qué píxeles están dentro de las bounding boxes.
        """
        height, width = image_shape[:2]
        grid = np.full((height, width), None)  # Inicializar la cuadrícula con None
        classes = np.full((height, width), None)  # Guardar las clases de los elementos
        for x, y, w, h, _, label, id in detections:
            grid[y:y + h, x:x + w] = id
            classes[y:y + h, x:x + w] = label
        return grid, classes
    
    def check_terminal_neighbors(cx, cy, visited, current_id, grid, classes, image_shape):
        """
        Explores the area around a terminal to find a nearby element ID that is different from the current ID.

        Args:
            cx (int): X-coordinate of the terminal.
            cy (int): Y-coordinate of the terminal.
            visited (set): Set of already visited coordinates.
            current_id (int or None): The ID of the current element being checked.
            grid (numpy.ndarray): A grid representing element IDs at each pixel.
            classes (numpy.ndarray): A grid representing element classes at each pixel.
            image_shape (tuple): Shape of the image as (height, width).

        Returns:
            tuple: A tuple containing the class and ID of a nearby element if found, otherwise (None, None).
        """
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited or not (0 <= nx < image_shape[1] and 0 <= ny < image_shape[0]):
                    continue
                element_id = grid[ny, nx]
                if element_id is not None and element_id != current_id:
                    element_class = classes[ny, nx]
                    return element_class, element_id
        return None, None

    def get_neighbors(x, y, image_shape):
        """
        Retrieves the neighboring coordinates of a given point within the image bounds.

        Args:
            x (int): X-coordinate of the point.
            y (int): Y-coordinate of the point.
            image_shape (tuple): Shape of the image as (height, width).

        Returns:
            list: List of valid neighboring coordinates as (nx, ny).
        """
        neighbors = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)
        ]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < image_shape[1] and 0 <= ny < image_shape[0]]
    
    def is_different_from_background(pixel, dominant_color):
        """
        Checks if a given pixel is different from the dominant background color.

        Args:
            pixel (numpy.ndarray): The pixel value as an array (e.g., [R, G, B]).
            dominant_color (numpy.ndarray): The dominant background color as an array (e.g., [R, G, B]).

        Returns:
            bool: True if the pixel is different from the dominant color, False otherwise.
        """
        return not np.array_equal(pixel, dominant_color)
    
    def flood_fill(x, y, visited, path, image, grid, classes, is_different_from_background, get_neighbors, check_terminal_neighbors, dominant_color):
        """
        Performs a flood-fill operation to explore connected regions in an image.

        Args:
            x (int): Starting X-coordinate.
            y (int): Starting Y-coordinate.
            visited (set): Set of already visited coordinates.
            path (list): List to store the coordinates of the explored path.
            image (numpy.ndarray): The image being processed.
            grid (numpy.ndarray): Grid representing element IDs at each pixel.
            classes (numpy.ndarray): Grid representing element classes at each pixel.
            is_different_from_background (function): Function to determine if a pixel differs from the background.
            get_neighbors (function): Function to retrieve valid neighboring coordinates.
            check_terminal_neighbors (function): Function to check for terminal connections near the current path.

        Returns:
            tuple: A tuple containing the detected element ID and class if found, otherwise (None, None).
        """
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited or not is_different_from_background(image[cy, cx], dominant_color):
                continue
            visited.add((cx, cy))
            path.append((cx, cy))
            inside_detection = grid[cy, cx]
            if inside_detection:
                return inside_detection, classes[cy, cx]
            stack.extend(get_neighbors(cx, cy, image.shape))

        if path:
            terminal_coords = [path[-1][0], path[-1][1]]
            terminal_label, terminal_id = check_terminal_neighbors(
                terminal_coords[0], terminal_coords[1], visited, None, grid, classes, image.shape
            )
            if terminal_label and terminal_id:
                return terminal_id, terminal_label

        return None, None
    
    def add_connections_to_svg(svg_path, output_folder, output_svg_name, connections):
        """
        Adds detected connections to an existing SVG file within a new layer.

        The new layer is called "connections" and is placed between the background
        and element_detections layers.

        Args:
            svg_path (str): Path to the original SVG file.
            output_folder (str): Folder to save the modified SVG file.
            output_svg_name (str): Name of the output SVG file.
            connections (list): List of connections in the format
                                [class_origin, id_origin, class_destination, id_destination, path].
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Folder created: {output_folder}")
    
        tree = ET.parse(svg_path)
        root = tree.getroot()
    
        background_layer = root.find(".//*[@id='background-layer']")
        element_detections_layer = root.find(".//*[@id='element_detections']")
    
        connections_layer = ET.Element("g", attrib={"id": "connections", "stroke": "orange", "stroke-width": "2", "fill": "none"})
    
        for conn in connections:
            if len(conn) >= 5 and isinstance(conn[4], list):
                path_points = conn[4]
                path_data = "M " + " L ".join(f"{x},{y}" for x, y in path_points)
                path_element = ET.Element("path", attrib={"d": path_data})
                connections_layer.append(path_element)
    
        if element_detections_layer is not None:
            position = list(root).index(element_detections_layer)
            root.insert(position, connections_layer)
        elif background_layer is not None:
            position = list(root).index(background_layer)
            root.insert(position + 1, connections_layer)
        else:
            root.append(connections_layer)
    
        output_svg_path = os.path.join(output_folder, output_svg_name)
        tree.write(output_svg_path, encoding="utf-8", xml_declaration=True)
        print(f"SVG with connections saved at {output_svg_path}")
        
    connections_results = []

    for image_pair in processed_images:
        if len(image_pair) < 3:
            print(f"Warning: Incorrect structure in processed_images: {image_pair}.")
            continue

        image_path, svg_path, detections = image_pair

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image could not be read at {image_path}. Skipping.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        dominant_color = np.array(get_dominant_color(image_path))
        grid, classes = precompute_grid(detections, image.shape)

        conexiones = []
        explored_connections = set()
        terminal_count = 0

        visited = set()
        for x, y, w, h, _, label, id in detections:
            start_pixels = []
            for i in range(x - 3, x + w + 4):
                if 0 <= i < image.shape[1]:
                    if 0 <= y - 3 < image.shape[0]:
                        start_pixels.append((i, y - 3))
                    if 0 <= y + h + 3 < image.shape[0]:
                        start_pixels.append((i, y + h + 3))
            for j in range(y - 3, y + h + 4):
                if 0 <= j < image.shape[0]:
                    if 0 <= x - 3 < image.shape[1]:
                        start_pixels.append((x - 3, j))
                    if 0 <= x + w + 3 < image.shape[1]:
                        start_pixels.append((x + w + 3, j))

            for start_x, start_y in start_pixels:
                if (start_x, start_y) in visited:
                    continue

                path = []
                result_id, result_label = flood_fill(start_x, start_y, visited, path, image, grid, classes, is_different_from_background, get_neighbors, check_terminal_neighbors, dominant_color)

                if result_id:
                    connection_pair = (id, result_id)
                    if connection_pair not in explored_connections:
                        explored_connections.add(connection_pair)
                        conexiones.append([label, id, result_label, result_id, path])
                else:
                    if path:
                        terminal_coords = [path[-1][0], path[-1][1]]
                        terminal_label, terminal_id = check_terminal_neighbors(terminal_coords[0], terminal_coords[1], visited, id, grid, classes, image.shape)
                        if terminal_label and terminal_id:
                            connection_pair = (id, terminal_id)
                            if connection_pair not in explored_connections:
                                explored_connections.add(connection_pair)
                                conexiones.append([label, id, terminal_label, terminal_id, path])
                        else:
                            terminal_count += 1
                            conexiones.append([label, id, terminal_coords, f"terminal_{terminal_count}", path])

        output_connections_png_path = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_connections_png_path, gray_image)

        add_connections_to_svg(svg_path, output_path, os.path.basename(svg_path), conexiones)

        connections_results.append([output_connections_png_path, svg_path, conexiones])
        
        print_amber("Connections done!")

    return connections_results


def main():
    print_amber("ReSin starting. Hello!")
    
    # OCR arguments
    input_path, workspace, ocr_output_path, ocr_language, ocr_confidence_threshold, template_library, template_output_path, template_confidence_threshold, iou_confidence_threshold, connections_output_path = ReSin_config()

    # OCR
    ocr_processed_images = text_detection(input_path, ocr_output_path, ocr_language, ocr_confidence_threshold)

    # Template matching
    template_processed_images = template_matching(ocr_processed_images, template_output_path, template_library, template_confidence_threshold, iou_confidence_threshold)

    # Connections
    connections = find_connections(template_processed_images, connections_output_path)

    print_amber("ReSin has finished its work. Done!")


# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()