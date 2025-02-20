"""
    Este módulo permite obtener los valores de la temperatura ponderada a partir de archivos de texto que se obtienen de 
    process_thermal.py.

    Recibe la ruta de la carpeta en donde se encuentran los archivos de texto con las temperatura de la nariz y retorna
    todas las temperaturas ponderadas a partir de un kernel gaussiano normalizado.
"""

import os
import numpy as np

def weighted_temperature_average(file_path, sigma=1.0):
    # Cargar la matriz de temperaturas desde el archivo de texto
    temp_matrix = np.loadtxt(file_path, delimiter=",")

    # Generar un kernel gaussiano del mismo tamaño que la matriz de temperatura
    kernel = np.exp(-((np.arange(temp_matrix.shape[0]) - temp_matrix.shape[0] // 2)[:, None]**2 + 
                       (np.arange(temp_matrix.shape[1]) - temp_matrix.shape[1] // 2)[None, :]**2) 
                    / (2 * sigma**2))
    
    # Normalizar el kernel para que la suma sea 1 (así mantiene el promedio correcto)
    kernel /= np.sum(kernel)

    # Aplicar el kernel multiplicando elemento a elemento
    weighted_sum = np.sum(temp_matrix * kernel)

    # Retornar el promedio ponderado
    return weighted_sum

# Aplicación
directory_path = "nose_temperatures"

with open('temperatures.txt', 'a') as archivo:
    for file in os.listdir(directory_path):
        # Verifica si es un archivo (ignora subdirectorios)
        if os.path.isfile(os.path.join(directory_path, file)):
            temp = weighted_temperature_average(os.path.join(directory_path, file), sigma=1)
            # Escribir el valor de temp en el archivo
            archivo.write(f'{temp}\n')