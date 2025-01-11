import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image_path):
    """
    Calcula el color predominante de una imagen.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen. Verifica la ruta.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    pixel_counts = Counter(map(tuple, pixels))
    sorted_colors = pixel_counts.most_common()
    for color, _ in sorted_colors:
        if color != (0, 0, 0):  # Ignorar el negro
            return color
    return (0, 0, 0)

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

def find_connections(image_path, detections, dominant_color):
    """
    Encuentra conexiones y terminales en una imagen a partir de detecciones de bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen. Verifica la ruta.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    dominant_color = np.array(dominant_color)
    grid, classes = precompute_grid(detections, image.shape)

    conexiones = []
    terminal_count = 0

    def is_different_from_background(pixel):
        return not np.array_equal(pixel, dominant_color)

    def get_neighbors(x, y):
        neighbors = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)
        ]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]]

    def flood_fill(x, y, visited, path):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited or not is_different_from_background(image[cy, cx]):
                continue
            visited.add((cx, cy))
            path.append((cx, cy))
            inside_detection = grid[cy, cx]
            if not inside_detection:
                gray_image[cy, cx] = [255, 165, 0]
            else:
                return inside_detection, classes[cy, cx]
            stack.extend(get_neighbors(cx, cy))
        return None, None

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
            result_id, result_label = flood_fill(start_x, start_y, visited, path)

            if result_id:  # Es una conexión
                conexiones.append([label, id, result_label, result_id, path])
            else:  # Es un terminal
                if path:
                    terminal_count += 1
                    terminal_coords = [path[-1][0], path[-1][1]]
                    conexiones.append([label, id, terminal_coords, f"terminal_{terminal_count}", path])

    return conexiones, gray_image

# Ejemplo de uso
image_path = "factory1/output_template/Sinoptico2_147.png"
detections = [(1291, 496, 60, 61, 0, '3way_valve', 4), (1532, 511, 129, 49, 90, 'column', 9), (923, 776, 45, 125, 0, 'column', 43), (1763, 956, 115, 35, 90, 'column', 45), (1223, 649, 61, 141, 0, 'column', 50), (1392, 111, 43, 41, 0, 'fan', 100), (312, 688, 43, 41, 0, 'fan', 101), (1015, 341, 124, 36, 180, 'flexible_duct', 104), (931, 503, 124, 36, 180, 'flexible_duct', 105), (1011, 416, 59, 58, 0, 'hopper', 110), (419, 261, 91, 86, 0, 'hopper', 111), (755, 258, 95, 90, 0, 'hopper', 112), (486, 685, 437, 205, 0, 'mixer', 148), (1349, 746, 91, 73, 0, 'pump', 153), (1338, 826, 91, 73, 0, 'pump', 157), (165, 247, 22, 77, 270, 'unk', 176), (614, 247, 22, 77, 270, 'unk', 177), (1209, 97, 67, 10, 0, 'unk', 187), (521, 191, 67, 10, 0, 'unk', 188), (857, 191, 67, 10, 0, 'unk', 189), (1616, 187, 73, 18, 0, 'unk', 214), (1076, 308, 73, 18, 0, 'unk', 216), (415, 416, 73, 18, 0, 'unk', 219), (752, 416, 73, 18, 0, 'unk', 222), (247, 496, 73, 18, 0, 'unk', 225), (415, 496, 73, 18, 0, 'unk', 228), (752, 496, 73, 18, 0, 'unk', 231), (908, 577, 73, 18, 0, 'unk', 233), (49, 767, 73, 18, 180, 'unk', 234), (208, 160, 34, 41, 0, 'valve', 235), (306, 160, 34, 41, 0, 'valve', 236), (651, 160, 34, 41, 0, 'valve', 237), (1500, 416, 34, 41, 0, 'valve', 238), (1542, 416, 34, 41, 0, 'valve', 239), (1584, 416, 34, 41, 0, 'valve', 240), (1626, 416, 34, 41, 0, 'valve', 241), (1668, 416, 34, 41, 0, 'valve', 242), (1738, 416, 34, 41, 0, 'valve', 243), (1780, 416, 34, 41, 0, 'valve', 244), (1822, 416, 34, 41, 0, 'valve', 245), (1865, 416, 34, 41, 0, 'valve', 246), (1233, 596, 34, 41, 0, 'valve', 247), (1584, 698, 34, 41, 0, 'valve', 248), (1801, 698, 34, 41, 0, 'valve', 249), (1422, 117, 152, 146, 0, 'washing_machine', 250), (160, 695, 152, 146, 0, 'washing_machine', 255)]

dominant_color = get_dominant_color(image_path)
result, highlighted_image = find_connections(image_path, detections, dominant_color)

for conn in result:
    if isinstance(conn[2], list):  # Si es un terminal
        cv2.circle(highlighted_image, tuple(conn[2]), 5, (0, 255, 0), -1)

cv2.imshow("Conexiones resaltadas", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
