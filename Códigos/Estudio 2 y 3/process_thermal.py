"""Este módulo permite extraer la zona de la nariz del archivo de cámaras térmicas .gzip de cada uno de los frames
   A partir de un archivo nose_detection.txt que contiene las coordenadas de la segmentación para la nariz.

   Retorna un archivo de texto por cada frame, en el que se encuentran los valores de temperaturas en grados celcius del sector 
   de la nariz en forma matricial.
"""

import os
import gzip
import numpy as np
import pandas as pd

# Configuración de archivos
frames_file = "DAF_11_M_240823_A.gzip"  # Archivo con los datos térmicos
coordinates_file_name = "nose_detections.txt"  # Archivo con las coordenadas
output_folder = "nose_temperatures"  # Carpeta para guardar los archivos de temperatura


def process_thermal_frames(filename, coordinates_file_name, output_folder):
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar coordenadas
    coordinates = pd.read_csv(coordinates_file_name, header=None, skiprows=1,
                          names=["image", "xmin", "ymin", "xmax", "ymax", "confidence"])
    
    with gzip.open(filename, 'rb') as f:
        # Leer framerate y tamaño de imagen
        framerate = np.frombuffer(f.read(8), dtype=float)
        size = np.frombuffer(f.read(4), dtype=np.uint16)
        w, h = size
        framelen = w * h * 2

        frame_index = 0

        while True:
            # Leer tiempo del frame
            buffer = f.read(8)
            if len(buffer) == 0:
                break
            
            timestamp = np.frombuffer(buffer, dtype=float)[0] # Marcador de tiempo

            # Leer los datos del frame
            buffer = f.read(framelen)
            if len(buffer) != framelen:
                break
            
            frame = np.frombuffer(buffer, dtype=np.uint16)
            frame.shape = (h, w)

            # Convertir a grados Celsius
            frame_celsius = (frame / 100.0) - 273.15

            # Se obtiene nombre de la imagen
            image_name = coordinates.iloc[frame_index]["image"]

            # Obtener coordenadas de la nariz
            xmin, ymin, xmax, ymax = map(int, coordinates.iloc[frame_index][["xmin", "ymin", "xmax", "ymax"]])

            # Extraer la región de la nariz
            nose_temp = frame_celsius[ymin:(ymax+1), xmin:(xmax+1)]

            # Guardar la matriz de temperaturas en un archivo .txt
            output_path = os.path.join(output_folder, f"{image_name[:-4]}.txt")
            np.savetxt(output_path, nose_temp, fmt="%.2f", delimiter=",")

            
            print(f"Guardado: {output_path}")

            frame_index += 1
            if frame_index >= len(coordinates):
                break  # No hay más coordenadas disponibles

# Ejecutar el procesamiento
if __name__ == "__main__":
    process_thermal_frames(frames_file, coordinates_file_name, output_folder)
