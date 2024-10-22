import cv2
import numpy as np
import easyocr
import svgwrite

# Variable global para el path de la imagen
IMAGE_PATH = 'img/img_test_2.png'

def create_svg_with_text(image_path, text_data):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Get the dimensions of the image (height, width, channels)
    height, width, channels = image.shape
    
    # Create a new SVG file
    svg_filename = image_path.replace('.png', '.svg')  # Cambiar la extensión a SVG
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

def text_detection(image_path):
    img = cv2.imread(image_path)

    # reader
    reader = easyocr.Reader(['en', 'es', 'de'], gpu=True)

    # Detect text
    text_data = reader.readtext(img)

    # Return detected text data for further use
    return text_data

def main():
    # Detect text and get bounding box information
    text_data = text_detection(IMAGE_PATH)
    
    # Create an SVG with text positioned based on bounding box coordinates
    create_svg_with_text(IMAGE_PATH, text_data)

# Llamar al main solo si se ejecuta como script principal
if __name__ == "__main__":
    main()