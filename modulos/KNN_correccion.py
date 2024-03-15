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

informacion_df = data_L_A.describe()

varianza_pixels = informacion_df.loc['std']

indices_mas_grandes = []

for i in range(3):
    indices_mas_grandes.append(varianza_pixels.idxmax())
    varianza_pixels = varianza_pixels.drop(varianza_pixels.idxmax())


X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label']

"""
Separar os datos en conjuntos de train y test.
d. Ajustar un modelo de KNN considerando pocos atributos, por ejemplo
3. Probar con distintos conjuntos de 3 atributos y comparar resultados.
Analizar utilizando otras cantidades de atributos.
"""


#Ahora separo en conjuntos de train y test, utilizando un 20% de los datos para 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,
                                                    shuffle=True, stratify= y)














