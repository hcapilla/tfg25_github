import os
import svgwrite
import easyocr
import logging

logging.disable(logging.WARNING)  # Desactiva los mensajes de advertencia y niveles inferiores

# Variable global para el path de la imagen
IMAGE_PATH = 'img/Sinoptico1.png'

def create_svg_with_text(image_path, text_data, suffix, text_threshold):
    # Load the image using OpenCV (only to get dimensions)
    import cv2
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Convert the threshold to a string (e.g., 0.9 -> '0_9')
    threshold_str = str(text_threshold).replace('.', '_')

    # Create a folder for the threshold
    img_folder = os.path.dirname(image_path)
    output_folder = os.path.join(img_folder, threshold_str)  # Directory named after the threshold value (e.g., '0_9')
    os.makedirs(output_folder, exist_ok=True)

    # Create the output SVG filename with the threshold value and image type
    img_name = os.path.basename(image_path).replace('.png', f'_{suffix}_{threshold_str}.svg')
    svg_filename = os.path.join(output_folder, img_name)

    # Create a new SVG file
    dwg = svgwrite.Drawing(svg_filename, size=(width, height))
    
    # Add a custom background color (RGB: (24, 24, 24) -> Hex: #181818)
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='#b5b2a8'))
    
    # Loop through the detected text data to position text on the SVG
    for bbox, text, score in text_data:
        # bbox contains 4 points (top-left, top-right, bottom-right, bottom-left)
        # Use the top-left corner for text positioning
        top_left = bbox[0]  # First point (top-left) of the bbox
        
        # Convert coordinates to integers
        x, y = int(top_left[0]), int(top_left[1])
        
        # Add the text at the top-left corner of the bounding box
        dwg.add(dwg.text(text, insert=(x, y), fill='black', font_size=20))

    # Save the SVG file
    dwg.save()
    
    # Print message
    print(f"SVG created: {svg_filename}")
    
    # Return the path of the created SVG
    return svg_filename

def text_detection(image_path, text_threshold=0.5):
    import cv2
    img = cv2.imread(image_path)

    # reader
    reader = easyocr.Reader(['en'], gpu=True)

    # Detect text
    text_data = reader.readtext(img)

    # Filter based on text_threshold
    filtered_text_data = [item for item in text_data if item[2] >= text_threshold]

    # Return detected text data for further use
    return filtered_text_data

def image_preprocessing(image_path):
    import cv2
    # Load the original image
    original_image = cv2.imread(image_path)

    # Convert to grayscale first
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Create a black and white version by applying threshold
    _, bw_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # Save the images in /img
    img_folder = os.path.dirname(image_path)
    base_name = os.path.basename(image_path).replace('.png', '')

    bw_image_path = os.path.join(img_folder, f"{base_name}.png")
    grayscale_image_path = os.path.join(img_folder, f"{base_name}.png")
    original_image_path = os.path.join(img_folder, f"{base_name}.png")

    cv2.imwrite(bw_image_path, bw_image)
    cv2.imwrite(grayscale_image_path, grayscale_image)
    cv2.imwrite(original_image_path, original_image)

    # Return paths to the generated images
    return original_image_path, bw_image_path, grayscale_image_path

def main():
    # Set the text thresholds (array of multiple values)
    text_thresholds = [0.7, 0.8, 0.9]

    # Preprocess the image to get original, black and white, and grayscale versions and save them
    original_image_path, bw_image_path, grayscale_image_path = image_preprocessing(IMAGE_PATH)

    # Store the images and thresholds for combining
    bw_svgs = []
    grayscale_svgs = []
    original_svgs = []
    all_thresholds = []

    # Loop through each threshold in the array
    for text_threshold in text_thresholds:
        # Detect text and create SVG for the original image
        text_data_original = text_detection(original_image_path, text_threshold)
        original_svg = create_svg_with_text(original_image_path, text_data_original, "raster", text_threshold)
        original_svgs.append(original_svg)

        # Detect text and create SVG for the black and white image
        text_data_bw = text_detection(bw_image_path, text_threshold)
        bw_svg = create_svg_with_text(bw_image_path, text_data_bw, "BW", text_threshold)
        bw_svgs.append(bw_svg)

        # Detect text and create SVG for the grayscale image
        text_data_grayscale = text_detection(grayscale_image_path, text_threshold)
        grayscale_svg = create_svg_with_text(grayscale_image_path, text_data_grayscale, "grayscale", text_threshold)
        grayscale_svgs.append(grayscale_svg)

        # Save the threshold value for each type
        all_thresholds.append(text_threshold)

# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()
