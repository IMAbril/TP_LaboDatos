#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:00:02 2024

@author: francisco
"""

import pandas as pd
import numpy as np
from graficos import sign, cantMuestras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random


# Cargar los datos desde el archivo CSV
carpeta = '/home/francisco/Documents/Labo de Datos/TP02/Archivos Python/'
data = pd.read_csv(carpeta+'sign_mnist_train (1).csv')

#%%
#A partir del dataframe original, construimos un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a las letras vocales.
data_vocales = sign[(sign['label'] == 0) | (sign['label'] == 4) | (sign['label'] == 8) | (sign['label'] == 13) | (sign['label'] == 19)]


#Obtuvimos la cantidad de muestras de cada letra al realizar la exploración.
cantidad_letra_A = cantMuestras[cantMuestras['letras'] == 'a']['cantidad'].values[0]
print('Cantidad de muestras de la letra A:', cantidad_letra_A)

# Iterar sobre todas las letras
for letra in ['E', 'I', 'O', 'U']:
    cantidad_letra = cantMuestras[cantMuestras['letras'] == letra.lower()]['cantidad'].values[0]
    diferencia_muestral = cantidad_letra_A / cantidad_letra
    print(f'La cantidad de muestras de la letra A es {diferencia_muestral} veces mayor que la cantidad de muestras de la letra {letra}.')

print('Las clases están balanceadas, pues al compararlas todas obtenemos un valor muy cercano a uno')
#%%
# Comenzamos separando nuestras variables 
X = data_vocales.drop(['label'], axis=1)
y = data_vocales['label']

#Separamos en un conjunto de entrenamiento y otro de validación. Colocamos una semilla, con random-state
# Separamos en 80% entrenamiento y 20% prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=10)

# Crear y ajustar modelos de árbol de decisión con diferentes profundidades
# Vamos a analizar profundidades de árbol de 1 a 15
depths = range(1, 16)
mean_scores = []

for depth in depths:
    # Crear el modelo de árbol de decisión
    clf = DecisionTreeClassifier(max_depth=depth, random_state=5)
    
    # Realizar validación cruzada con k-folding (k = 5)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    # Calcular el promedio de los puntajes de validación cruzada
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)
    
    print(f"Profundidad: {depth}, Precisión Media: {mean_score}")

# Seleccionar la mejor profundidad del árbol
best_depth_index = np.argmax(mean_scores)
best_depth = depths[best_depth_index]
print(f"Mejor Profundidad: {best_depth}")

# Entrenar el modelo de árbol de decisión con la mejor profundidad
best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=10)
best_clf.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en el conjunto de test: {test_accuracy}")

#%%
# Graficar la precisión promedio en validación cruzada para cada profundidad
plt.figure(figsize=(8, 5))
plt.plot(depths, mean_scores, marker='o', color='green')
plt.title('Precisión Media en Validación Cruzada vs. Profundidad del Árbol')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Precisión Media')
plt.xticks(depths)
plt.grid(True)
plt.show()


#%%
# Genera las predicciones y la matriz de confusión
y_pred = best_clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

# Crea un DataFrame de pandas para la matriz de confusión
conf_mat_df = pd.DataFrame(conf_mat, index=['A', 'E', 'I', 'O', 'U'], columns=['A', 'E', 'I', 'O', 'U'])

# Crea el gráfico de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_df, annot=True, fmt='d', cmap='coolwarm', cbar=True, linewidths=0.5, linecolor='black')
plt.title('Matriz de Confusión', fontsize=20)
plt.xlabel('Predicciones', fontsize=15)
plt.ylabel('Valores Reales', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.show()


