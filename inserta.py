import xml.etree.ElementTree as ET
from xml.dom import minidom

def limpiar_namespaces(element):
    """Remueve el namespace de los elementos XML."""
    for elem in element.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # Remover el namespace

def insertar_svg(lienzo_path, insertar_path, output_path, bounding_box):
    """Inserta un SVG dentro de una bounding box definida como 4 puntos."""
    # Cargar el lienzo principal
    tree_lienzo = ET.parse(lienzo_path)
    root_lienzo = tree_lienzo.getroot()

    # Limpiar namespaces del lienzo
    limpiar_namespaces(root_lienzo)

    # Cargar el SVG a insertar
    tree_insertar = ET.parse(insertar_path)
    root_insertar = tree_insertar.getroot()

    # Limpiar namespaces del SVG a insertar
    limpiar_namespaces(root_insertar)

    # Calcular bounding box a partir de 4 puntos
    bbox_x = bounding_box[0]
    bbox_y = bounding_box[1]
    bbox_width = bounding_box[2] - bounding_box[0]
    bbox_height = bounding_box[3] - bounding_box[1]

    # (Resto del código permanece igual...)
    # Obtener dimensiones físicas del SVG a insertar
    width_in = float(root_insertar.attrib.get('width', '0').replace('in', ''))
    height_in = float(root_insertar.attrib.get('height', '0').replace('in', ''))
    width_px = width_in * 96  # Convertir pulgadas a píxeles
    height_px = height_in * 96

    # Obtener el viewBox del SVG a insertar
    viewBox = root_insertar.attrib.get('viewBox', '0 0 1 1').split()
    viewBox_width = float(viewBox[2])
    viewBox_height = float(viewBox[3])

    # Calcular el factor de escala para ajustar el viewBox a las dimensiones físicas
    scale_viewBox_x = width_px / viewBox_width
    scale_viewBox_y = height_px / viewBox_height

    # Calcular el factor de escala uniforme para que el SVG quepa en la bounding box
    scale_x = bbox_width / width_px
    scale_y = bbox_height / height_px
    scale = min(scale_x, scale_y)  # Escala uniforme para evitar distorsión

    # Calcular el offset para centrar el SVG en la bounding box
    offset_x = bbox_x + (bbox_width - (width_px * scale)) / 2
    offset_y = bbox_y + (bbox_height - (height_px * scale)) / 2

    # Crear un grupo con las transformaciones
    grupo = ET.Element('g', attrib={
        'transform': f'translate({offset_x},{offset_y}) scale({scale * scale_viewBox_x}, {scale * scale_viewBox_y})'
    })

    # Insertar los elementos del SVG dentro del grupo
    for elem in root_insertar:
        grupo.append(elem)

    # Agregar el grupo al lienzo principal
    root_lienzo.append(grupo)

    # Guardar el resultado con formato UTF-8
    xmlstr = minidom.parseString(ET.tostring(root_lienzo, encoding='unicode')).toprettyxml(indent="   ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xmlstr)

    print(f"SVG insertado y guardado en: {output_path}")

if __name__ == "__main__":
    # Configuración directa dentro del script
    svg_insertar = "mixer.svg"  # Archivo SVG a insertar
    svg_base = "sinoptico.svg"  # Archivo SVG base
    output = "sinoptico_output.svg"  # Archivo de salida
    bounding_box = [100, 100, 120, 200]
    insertar_svg("sinoptico.svg", "mixer.svg", "sinoptico_output.svg", bounding_box)

