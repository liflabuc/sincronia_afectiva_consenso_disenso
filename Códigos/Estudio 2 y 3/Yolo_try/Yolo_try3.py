import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ruta de la imagen generada en try2
image_path = r"G:\My Drive\2024\ICESI\Lab_Chile_NOMODIFICAR\MIA_SI_MODIFICAR\fotogramas\persona_cabeza_0876.jpg"

# Cargar la imagen en escala de grises
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar un filtro de desenfoque para suavizar la imagen y reducir el ruido
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Aplicar un umbral adaptativo para resaltar las áreas oscuras
_, binary_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)

# Encontrar los contornos en la imagen binarizada
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos para encontrar la nariz
nose_contour = None
image_center = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
min_distance = float('inf')

nose_contour = None
min_cY = float('inf')  # Inicializa con un valor muy grande

for contour in contours:
    # Aproximar el contorno a un polígono 
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if 3 <= len(approx) < 12: # Ajustar Num vertices 
        # Calcular el centro del contorno
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Verificar si el centro del contorno está cerca del centro de la imagen
            distance = np.sqrt((cX - image_center[0]) ** 2 + (cY - image_center[1]) ** 2)
            if distance < gray_image.shape[1] // 4:
                # Verificar si el área del contorno es significativa
                area = cv2.contourArea(contour)
                if area < 2000:  # Ajusta tamaño esperado de la nariz
                    # Seleccionar el contorno más cercano a la parte superior de la imagen (menor cY) - evita mentón
                    if cY < min_cY:
                        min_cY = cY
                        nose_contour = contour
                        
# Crear una copia de la imagen original para dibujar encima el contorno
output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
if nose_contour is not None:
    cv2.drawContours(output_image, [nose_contour], -1, (0, 255, 0), 2)
else:
    print("No se encontró una región adecuada en la imagen.")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Región de la nariz (resaltada en verde)')
plt.axis('off')
plt.show()

# Guardar la imagen con el contorno 
output_path = r"G:\My Drive\2024\ICESI\Lab_Chile_NOMODIFICAR\MIA_SI_MODIFICAR\fotogramas\persona_cabeza_nariz_0876_2.jpg"
cv2.imwrite(output_path, output_image)
print(f"Imagen nariz guardada en {output_path}")
