# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024
"""
from modulos.exploracion import  cantMuestras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import random 
import numpy as np 
import pandas as pd 

carpeta = '/Users/Roju2/OneDrive/Desktop/'
sign = pd.read_csv(carpeta +'sign_mnist_train.csv')

#A partir del dataframe original, construimos un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a señas de las letras L o A.
data_L_A = sign[(sign['label'] == 0) | (sign['label'] == 11)]

def muestras():    
    
    
    #Obtuvimos la cantidad de muestras de cada letra al realizar la exploración.
    cantidad_letra_A = cantMuestras[cantMuestras['letras'] == 'a']['cantidad'].values[0]
    print('Cantidad de muestras de la letra A:', cantidad_letra_A)
    
    cantidad_letra_L = cantMuestras[cantMuestras['letras'] == 'l']['cantidad'].values[0]
    print('Cantidad de muestras de la letra L:', cantidad_letra_L)
    
    diferenciaMuestral = cantidad_letra_A/cantidad_letra_L
    print('Las clases están balanceadas, pues al compararlas obtenemos un valor muy cercano a uno : ', round(diferenciaMuestral,2))

#%%
#Para probar con distintos conjuntos de tres atributos, teniendo en cuenta la variabilidad pixel a pixel entre las muestras de cada letra
#Vamos a considerar aquellos con mayor, menor y variabilidad intermedia 

informacion_df = data_L_A.describe()

varianza_pixels = informacion_df.loc['std']

#Buscamos los mas grandes

indices_mas_grandes = []

serieMax = varianza_pixels.copy()

for j in range(3):
    indices_mas_grandes.append(serieMax.idxmax())
    serieMax = serieMax.drop(serieMax.idxmax())  # Elimina el mínimo y actualiza la Serie
print(indices_mas_grandes)

#Buscamos los mas pequeños
indices_mas_pequeños = []

serieMin = varianza_pixels.copy()
serieMin = serieMin.drop('label')  # Elimina la etiqueta 'label' y actualiza la Serie

for j in range(3):
    indices_mas_pequeños.append(serieMin.idxmin())
    
    serieMin = serieMin.drop(serieMin.idxmin())  # Elimina el mínimo y actualiza la Serie

print(indices_mas_pequeños)

#Buscamos los intermedios 

serieMed =varianza_pixels.copy()
#serieMed =serieMed.drop('label')
indices_medianos = []
for k in range(3):
    # Calcula la mediana
    mediana = np.median(serieMed)
    for clave, valor in serieMed.items():
        if valor == mediana:
            indices_medianos.append(clave)
    
    serieMed = serieMed.drop(clave)

print(indices_medianos)














