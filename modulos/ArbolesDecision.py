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
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
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

# Separamos en un conjunto de entrenamiento y otro de validación. Colocamos una semilla, con random-state
# Separamos en 80% train y 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=5)

arbol = DecisionTreeClassifier()
arbol.fit(X_train, y_train)# Crear y ajustar modelos de árbol de decisión con diferentes profundidades
# Vamos a analizar profundidades de árbol de 1 a 15
depths = range(1, 16)
mean_scores = []

for depth in depths:
    # Creamos el modelo de árbol de decisión
    arbol = DecisionTreeClassifier(max_depth=depth, random_state=5)
    # Hacemos validación cruzada con k-folding (k = 5)
    scores = cross_val_score(arbol, X_train, y_train, cv=5)
    # Calculamos el promedio de los puntajes de validación cruzada
    mean_score = np.mean(scores)
    mean_scores.append(mean_score)

    
    print(f"Profundidad: {depth}, Precisión Media: {mean_score}")
    

#%%

resultados_modelos = []
# Evaluamos cada modelo en el conjunto de test
for modelo in depths:
    # Entrenamos los arboles, con las mismas alturas y semilla para obtener los mismos arboles que antes
    arbol = DecisionTreeClassifier(max_depth=depth, random_state=5)
    arbol.fit(X_train, y_train)
    # Generamos las predicciones en el conjunto de test
    y_predict_test = arbol.predict(X_test)
    
    # Calculamos la precisión del modelo en el conjunto de test
    accuracy_test = accuracy_score(y_test, y_predict_test)
    
    # Calculamos el informe de clasificación en el conjunto de test
    classification_rep_test = classification_report(y_test, y_predict_test)
    
    # Calculamos la matriz de confusión en el conjunto de test
    conf_mat_test = confusion_matrix(y_test, y_predict_test)
    
    # Guardamos las métricas del modelo en una lista
    resultados_modelos.append({
        'modelo': arbol,
        'precision': accuracy_test,
        'informe_clasificacion': classification_rep_test,
        'matriz_confusion': conf_mat_test
    })

# Buscamos el modelo con la mejor precisión en el conjunto de test
mejor_modelo = max(resultados_modelos, key=lambda x: x['precision'])

profundidad_mejor_modelo = mejor_modelo['modelo'].max_depth
print("Profundidad del mejor modelo:", profundidad_mejor_modelo)

# Imprimimos las métricas del mejor modelo para saber todos sus datos
print(f"Precisión en conjunto de test: {mejor_modelo['precision']}")
print("Informe de clasificación en conjunto de test:")
print(mejor_modelo['informe_clasificacion'])
print("Matriz de confusión en conjunto de test:")
print(mejor_modelo['matriz_confusion'])


#%%
# Graficamos la precisión promedio en validación cruzada para cada profundidad
plt.figure(figsize=(8, 5))
plt.plot(depths, mean_scores, marker='o', color='green')
plt.title('Precisión Media en Validación Cruzada vs. Profundidad del Árbol')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Precisión Media')
plt.xticks(depths)
plt.grid(True)
plt.show()

#%%
# Creamos un DataFrame de pandas para la matriz de confusión
conf_mat_df = pd.DataFrame(conf_mat_test, index=['A', 'E', 'I', 'O', 'U'], columns=['A', 'E', 'I', 'O', 'U'])

# Creamos el gráfico de la matriz de confusión
plt.figure(figsize=(12, 6))
sns.heatmap(conf_mat_df, annot=True, fmt='d', cmap='coolwarm', cbar=True, linewidths=0.5, linecolor='black')
plt.title('Matriz de Confusión', fontsize=20)
plt.xlabel('Predicciones', fontsize=15)
plt.ylabel('Valores Reales', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.show()

