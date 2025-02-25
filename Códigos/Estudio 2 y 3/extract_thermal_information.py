"""
Este módulo fue creado para el procesamiento de frames obtenidos a partir de videos de cámaras térmicas.

Retorna dos archivos TXT. El primero, contiene la información de cada frame: Nombre del frame, timestamp, datos de temperatura en la ROI y temperatura ponderada.
El segundo es una lista de los frames que no están en el archivo de coordenadas.

Para utilizarlo, ejecutar en consola:

    python extract_thermal_information.py <nombre_archivo_thermal.gzip> <nombre_archivo_coordenadas.txt>


"""
import sys
import gzip
import numpy as np
import pandas as pd


def extract_thermal_frames_information(file_name: str):
    """
    Recibe un archivo GZIP correspondiente a una grabación con camáras térmicas.

    Retorna un generador de listas en que cada lista contiene la información
    del frame: Nombre (str), Marcador de tiempo (float), píxeles del frame (en Kelvin) (numpy array)
    """
    with gzip.open(file_name, 'rb') as file_gzip:
        # Leer framerate y tamaño de los frames
        framerate = np.frombuffer(file_gzip.read(8), dtype=float)
        size = np.frombuffer(file_gzip.read(4), dtype=np.uint16)
        print(f"Frame rate: {framerate}, Size: {size}")

        # Verificar lectura del tamaño
        if size.size < 2:
            print("No se pudo leer el tamaño de los frames correctamente.")
            return

        # Definición de parámetros
        w, h = size
        framelen = w * h * 2
        index_frame = 0

        while True:

            # Leer marcador de tiempo en el frame
            buffer = file_gzip.read(8)
            if not buffer:
                print("No hay más datos en el archivo GZIP.")
                break
            timestamp = float(np.frombuffer(buffer, dtype=np.float64)[0])

            # Leer pixeles del frame
            buffer = file_gzip.read(framelen)
            frame = np.frombuffer(buffer, dtype=np.uint16)
            frame.shape = (h, w)
            # frame = [";".join(map(str, fila)) for fila in frame]

            # Asignar nombre al frame, según el índice correspondiente
            frame_name = f"{file_name[:17]}_frame{str(index_frame).zfill(4)}.png"
            index_frame += 1

            yield [frame_name, timestamp, frame]


# Argumentos por consola
script_name = sys.argv[0]
file_name = sys.argv[1]
coordinates_file_name = sys.argv[2]

# Procesamiento de frames
if __name__ == "__main__":
    coordinates = pd.read_csv(coordinates_file_name, header=None, skiprows=1,
                              names=["image", "xmin", "ymin", "xmax", "ymax", "confidence"])

    with open("thermal_frame_information.txt", "w") as file_information:
        file_information.write("image,timestamp,roi_temp,weighted_temp\n")

        with open("no_faces_information.txt", "w") as file_no_faces:

            for information in extract_thermal_frames_information(file_name):
                frame_name = information[0]
                timestamp = information[1]
                frame = information[2]

                if information[0] in coordinates["image"].values:

                    # Extraer la región de interés, en Kelvin
                    xmin, ymin, xmax, ymax = coordinates.loc[coordinates["image"] == frame_name, [
                        "xmin", "ymin", "xmax", "ymax"]].values[0]
                    roi_temp = frame[ymin:(ymax+1), xmin:(xmax+1)]
                    roi_temp = roi_temp/100

                    # Generar un kernel normalizado y ponderar las temperaturas
                    sigma = 1.0
                    kernel = np.exp(-((np.arange(roi_temp.shape[0]) - roi_temp.shape[0] // 2)[:, None]**2 + (
                        np.arange(roi_temp.shape[1]) - roi_temp.shape[1] // 2)[None, :]**2) / (2 * sigma**2))
                    kernel /= np.sum(kernel)
                    weighted_sum = np.sum(roi_temp * kernel)

                    # Formato de temperaturas para el archivo TXT
                    roi_temp_str = "[" + ";".join(" ".join(map(str, row))
                                                  for row in roi_temp) + "]"
                    # Escribir la información en archivo TXT
                    file_information.write(
                        f"{frame_name},{timestamp},{roi_temp_str},{weighted_sum}\n")

                else:
                    # Escribir la información en archivos TXT
                    file_information.write(f"{frame_name}\n")
                    file_no_faces.write(f"{frame_name}\n")
