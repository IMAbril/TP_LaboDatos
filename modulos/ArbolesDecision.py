#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024

"""

import pandas as pd
import numpy as np
from modulos.exploracion import sign, cantMuestras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import random


#%%
# A partir del dataframe original, construimos un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a las letras vocales.
data_vocales = sign[(sign['label'] == 0) | (sign['label'] == 4) | (sign['label'] == 8) | (sign['label'] == 13) | (sign['label'] == 19)]

def vocales():
        # Obtuvimos la cantidad de muestras de cada letra al realizar la exploración.
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

depths = range(1, 16)

def profundidad_arbol():
        
    # Separamos en un conjunto de entrenamiento y otro de validación. Colocamos una semilla, con random-state
    # Separamos en 90% dev y 10% eval
    X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, shuffle=True, test_size=0.1, random_state=5)
    
    # Creamos y ajustamos modelos de árbol de decisión con diferentes profundidades
    # Vamos a analizar profundidades de árbol de 1 a 15
    mejor_modelo = None
    mejor_mean_score = 0
    mean_score = 0
    arboles = {}
    print("---------------K-FOLD Cross Validation en datos de dev---------------")
    for depth in depths:
        #guardo el mean_score anterior
        mean_score_anterior = mean_score
        # Creamos el modelo de árbol de decisión
        arbol = DecisionTreeClassifier(max_depth=depth, random_state=5)
        # Hacemos validación cruzada con k-folding (k = 5)
        scores = cross_val_score(arbol, X_dev, y_dev, cv=5)
        # Calculamos el promedio de los puntajes de validación cruzada
        mean_score = np.mean(scores)       
        #Guardamos los modelos
        arboles[depth] = arbol
        #Calculamos la diferencia del mean score actual con el anterior
        diferencia = mean_score - mean_score_anterior
        #Guardamos el modelo de mejor score
        if mean_score > mejor_mean_score and diferencia > 0.03 :
            mejor_modelo = arbol
            mejor_mean_score = mean_score
        
        print(f"Profundidad: {depth}, Precisión Media: {mean_score}, Diferencia: {diferencia}")

    # Nos decantamos por el modelo con altura 8 y lo evaluamos en los datos de eval
    print("---------------Validación del mejor modelo en datos nuevos (Eval)---------------")
    mejor_modelo.fit(X_dev,y_dev)
    y_pred = mejor_modelo.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)
    print(f"Mejor modelo: {mejor_modelo}, Exactitud:{accuracy} ")

    #Calculamos el informe de clasificación en el conjunto de eval o test
    classification_rep_test = classification_report(y_eval, y_pred)

    # Calculamos la matriz de confusión en el conjunto de eval o test
    conf_mat_test = confusion_matrix(y_eval, y_pred)

    # Guardamos las métricas del modelo en un diccionario
    reporte = {
        'modelo': arbol,
        'exactitud': accuracy,
        'informe_clasificacion': classification_rep_test,
        'matriz_confusion': conf_mat_test
        }
    # Imprimimos las métricas del mejor modelo para saber todos sus datos
    print(f"Exactitud en conjunto de test: {reporte['exactitud']}")
    print("Informe de clasificación en conjunto de test:")
    print(reporte['informe_clasificacion'])
    print("Matriz de confusión en conjunto de test:")
    print(reporte['matriz_confusion'])
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

    
    # Métricas para cada clase extraidas del informe de clasificación
    precision = [0.94, 0.91, 0.95, 0.84, 0.95]
    recall = [0.96, 0.91, 0.86, 0.95, 0.91]
    f1_score = [0.95, 0.91, 0.90, 0.89, 0.93]
    
    # Definir las etiquetas para las clases
    labels = ['A', 'E', 'I', 'O', 'U']
    
    # Crear un array de índices para las barras
    x = np.arange(len(labels))
    
    # Definir el ancho de las barras
    width = 0.25
    
    # Crear el gráfico de barras apiladas
    fig, ax = plt.subplots()
    
    ax.bar(x, precision, width, label='Precision')
    ax.bar(x + width, recall, width, label='Recall')
    ax.bar(x + 2*width, f1_score, width, label='F1-Score')
    
    # Añadir etiquetas, título y leyenda
    ax.set_xlabel('Clases')
    ax.set_ylabel('Métricas')
    ax.set_title('Métricas de Clasificación por Clase')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylim(0.8,1)
    plt.show()

    # print("---------------Validación de cada modelo en datos nuevos (eval)")
    # #Evaluamos cada uno de los modelos utilizando el conjunto de eval.
    # for depth, arbol in arboles.items():
    #     arbol.fit(X_dev,y_dev)
    #     y_p = arbol.predict(X_eval)
    #     acc = accuracy_score(y_eval, y_p)
    #     print(f"Profundidad: {depth}, Exactitud: {acc}")
    

