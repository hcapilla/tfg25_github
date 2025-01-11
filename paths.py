import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image_path):
    """
    Calcula el color predominante de una imagen.
    Si el color predominante es negro, retorna el segundo color más frecuente.

    Args:
        image_path (str): Ruta de la imagen.

    Returns:
        tuple: Color predominante en formato (R, G, B).
    """
    # Leer la imagen
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("No se pudo leer la imagen. Verifica la ruta.")

    # Convertir la imagen a formato RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplanar la matriz para obtener los colores en formato (R, G, B)
    pixels = image.reshape(-1, 3)

    # Contar la frecuencia de cada color
    pixel_counts = Counter(map(tuple, pixels))

    # Ordenar por frecuencia de mayor a menor
    sorted_colors = pixel_counts.most_common()

    # Buscar el color predominante que no sea negro
    for color, _ in sorted_colors:
        if color != (0, 0, 0):  # Ignorar el negro
            return color

    # Si todos los colores son negros, retornar negro
    return (0, 0, 0)

def find_connections(image_path, detections, dominant_color):
    """
    Encuentra conexiones y terminales en una imagen a partir de detecciones de bounding boxes.

    Args:
        image_path (str): Ruta de la imagen.
        detections (list): Lista de detecciones con formato (x, y, w, h, rot, label, tipo).
        dominant_color (tuple): Color de fondo dominante a ignorar (R, G, B).

    Returns:
        list: Lista de conexiones y terminales.
        numpy.ndarray: Imagen con las conexiones resaltadas.
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen. Verifica la ruta.")

    # Convertir la imagen a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Convertir a 3 canales para superposición

    # Convertir el color dominante a un array numpy
    dominant_color = np.array(dominant_color)

    conexiones = []
    terminal_count = 0

    def is_different_from_background(pixel):
        """Verifica si un pixel es diferente al color de fondo."""
        return not np.array_equal(pixel, dominant_color)

    def is_inside_detection(cx, cy):
        """Verifica si un píxel está dentro de alguna bounding box."""
        for dx, dy, dw, dh, _, _, _ in detections:
            if dx <= cx <= dx + dw and dy <= cy <= dy + dh:
                return True
        return False

    def get_neighbors(x, y):
        """Obtiene los vecinos de un píxel."""
        neighbors = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)
        ]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]]

    def flood_fill(x, y, visited, path):
        """Realiza un flood-fill para encontrar una conexión o terminal."""
        stack = [(x, y)]
        steps_since_last_check = 0
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited or not is_different_from_background(image[cy, cx]):
                continue
            visited.add((cx, cy))
            path.append((cx, cy))

            # Depurar: Verificar si el píxel está dentro de una bounding box
            inside_detection = is_inside_detection(cx, cy)
            #print(f"Pixel: ({cx}, {cy}), Inside detection: {inside_detection}")

            # Marcar píxel en naranja brillante si no está dentro de una bounding box
            if not inside_detection:
                gray_image[cy, cx] = [255, 165, 0]
            else:  # Opcional: marcar píxeles dentro de detecciones en azul
                gray_image[cy, cx] = [0, 0, 255]

            # Verificar cada 10 pasos si se conecta con otra detección
            steps_since_last_check += 1
            if steps_since_last_check == 100000000000000000000001000000000000000000000010000000000000000000000100000000000000000000001000000000000000000000010000000000000000000000:
                steps_since_last_check = 0
                for det in detections:
                    dx, dy, dw, dh, _, _, _ = det
                    if dx <= cx <= dx + dw and dy <= cy <= dy + dh:
                        return det

            stack.extend(get_neighbors(cx, cy))
        return None

    # Procesar cada detección
    visited = set()
    for idx, det in enumerate(detections):
        x, y, w, h, _, label, id = det
        start_x, start_y = x + w // 2, y + h // 2

        if (start_x, start_y) in visited:
            continue

        path = []
        result = flood_fill(start_x, start_y, visited, path)

        if result:
            conexiones.append((f"{label}_{id}", result[5]))  # Conexión encontrada
        else:
            terminal_count += 1
            conexiones.append(({label},{id}, "terminal", {terminal_count}))  # Terminal encontrada

    return conexiones, gray_image

# Ejemplo de uso
image_path = "factory1/output_ocr/Sinoptico2_147.png"
detections = [(1291, 496, 60, 61, 0, '3way_valve', 4), (1532, 511, 129, 49, 90, 'column', 9), (923, 776, 45, 125, 0, 'column', 43), (1763, 956, 115, 35, 90, 'column', 45), (1223, 649, 61, 141, 0, 'column', 50), (1392, 111, 43, 41, 0, 'fan', 100), (312, 688, 43, 41, 0, 'fan', 101), (1015, 341, 124, 36, 180, 'flexible_duct', 104), (931, 503, 124, 36, 180, 'flexible_duct', 105), (1011, 416, 59, 58, 0, 'hopper', 110), (419, 261, 91, 86, 0, 'hopper', 111), (755, 258, 95, 90, 0, 'hopper', 112), (486, 685, 437, 205, 0, 'mixer', 148), (1349, 746, 91, 73, 0, 'pump', 153), (1338, 826, 91, 73, 0, 'pump', 157), (165, 247, 22, 77, 270, 'unk', 176), (614, 247, 22, 77, 270, 'unk', 177), (1209, 97, 67, 10, 0, 'unk', 187), (521, 191, 67, 10, 0, 'unk', 188), (857, 191, 67, 10, 0, 'unk', 189), (1616, 187, 73, 18, 0, 'unk', 214), (1076, 308, 73, 18, 0, 'unk', 216), (415, 416, 73, 18, 0, 'unk', 219), (752, 416, 73, 18, 0, 'unk', 222), (247, 496, 73, 18, 0, 'unk', 225), (415, 496, 73, 18, 0, 'unk', 228), (752, 496, 73, 18, 0, 'unk', 231), (908, 577, 73, 18, 0, 'unk', 233), (49, 767, 73, 18, 180, 'unk', 234), (208, 160, 34, 41, 0, 'valve', 235), (306, 160, 34, 41, 0, 'valve', 236), (651, 160, 34, 41, 0, 'valve', 237), (1500, 416, 34, 41, 0, 'valve', 238), (1542, 416, 34, 41, 0, 'valve', 239), (1584, 416, 34, 41, 0, 'valve', 240), (1626, 416, 34, 41, 0, 'valve', 241), (1668, 416, 34, 41, 0, 'valve', 242), (1738, 416, 34, 41, 0, 'valve', 243), (1780, 416, 34, 41, 0, 'valve', 244), (1822, 416, 34, 41, 0, 'valve', 245), (1865, 416, 34, 41, 0, 'valve', 246), (1233, 596, 34, 41, 0, 'valve', 247), (1584, 698, 34, 41, 0, 'valve', 248), (1801, 698, 34, 41, 0, 'valve', 249), (1422, 117, 152, 146, 0, 'washing_machine', 250), (160, 695, 152, 146, 0, 'washing_machine', 255)]

dominant_color = get_dominant_color(image_path)
result, highlighted_image = find_connections(image_path, detections, dominant_color)

# Mostrar la imagen con los píxeles resaltados
cv2.imshow("Conexiones resaltadas", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result)
