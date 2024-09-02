from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo de segmentación YOLOv8 preentrenado
model = YOLO("yolov8n-seg.pt")  

# Ruta frame
image_path = r"G:\My Drive\2024\ICESI\Lab_Chile_NOMODIFICAR\MIA_SI_MODIFICAR\fotogramas\frame_0109.jpg"

# Segmentación en la imagen
results = model(image_path, task='segment')

# Filtrar persona
for result in results:
    # Verificar si hay persona 
    if result.masks is not None and len(result.masks.data) > 0:
        for i, class_id in enumerate(result.boxes.cls):
            if class_id == 0:  # 0 es persona
                # Obtener la máscara correspondiente
                mask = result.masks.data[i].cpu().numpy()

                # Convertir a imagen binaria
                mask_image = (mask * 255).astype(np.uint8)

                # Cargar la imagen original
                original_image = cv2.imread(image_path)

                # Recortar la imagen original usando la máscara
                segmented_person = cv2.bitwise_and(original_image, original_image, mask=mask_image)

                # Encontrar el contorno de la máscara para recortar 
                x, y, w, h = cv2.boundingRect(mask_image)

                # Determinar la región del cuello hacia arriba
                head_height = int(h * 0.4)  # Ajustable, está en 40% 
                cropped_head = segmented_person[y:y+head_height, x:x+w]

                # Guardar la imagen de la cabeza
                output_path = r"G:\My Drive\2024\ICESI\Lab_Chile_NOMODIFICAR\MIA_SI_MODIFICAR\fotogramas\persona_cabeza_0109.jpg"
                cv2.imwrite(output_path, cropped_head)
                print(f"Segmentación guardada en {output_path}")
    else:
        print("No se detectó ninguna persona en la imagen.")
