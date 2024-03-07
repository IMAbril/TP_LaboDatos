# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024
"""
from exploracion import sign
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Trabajo Práctico N°2
#A partir del dataframe original, construimos un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a señas de las letras L o A.

data_L_A = sign[(sign['label'] == 0) | (sign['label'] == 11)]