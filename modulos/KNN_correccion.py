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

def get_indices_mas_grandes():        
        serieMax = varianza_pixels.copy()
        
        for j in range(3):
            indices_mas_grandes.append(serieMax.idxmax())
            serieMax = serieMax.drop(serieMax.idxmax())  # Elimina el maximo y actualiza la Serie
        return indices_mas_grandes
get_indices_mas_grandes() 
#Buscamos los mas pequeños
indices_mas_chicos = []

def get_indices_mas_chicos():
    serieMin = varianza_pixels.copy()
    serieMin = serieMin.drop('label')  # Elimina la etiqueta 'label' y actualiza la Serie
   
    for j in range(3):                
        indices_mas_chicos.append(serieMin.idxmin())    
        serieMin = serieMin.drop(serieMin.idxmin())  # Elimina el mínimo y actualiza la Serie
    return indices_mas_chicos
get_indices_mas_chicos()
#Buscamos los intermedios
#Para ello, ordenamos, buscamos la mediana, el siguiente y el anterior 
indices_medianos = []
def get_indices_medianos():
    serieMed =varianza_pixels.copy()
    serieMed = pd.DataFrame(serieMed)     # Convertir a DataFrame
    serieMed_ordenado = serieMed.sort_values(by='std') #ordenamos de acuerdo al 'std'

    mediana_index = len(serieMed_ordenado) // 2 #calculamos mediana
    mediana = serieMed_ordenado.iloc[mediana_index]['std']
    indices_medianos.append( serieMed_ordenado.index[mediana_index])
    # Obtener el nombre del índice anterior a la mediana
    indice_anterior = serieMed_ordenado.index[mediana_index - 1] if mediana_index > 0 else None
    indices_medianos.append(indice_anterior)
    # Obtener el nombre del índice siguiente a la mediana
    indice_siguiente = serieMed_ordenado.index[mediana_index + 1] if mediana_index < len(serieMed_ordenado) - 1 else None
    indices_medianos.append(indice_siguiente)
    return indices_medianos

get_indices_medianos()


#%%


"""
Separar os datos en conjuntos de train y test.
d. Ajustar un modelo de KNN considerando pocos atributos, por ejemplo
3. Probar con distintos conjuntos de 3 atributos y comparar resultados.
Analizar utilizando otras cantidades de atributos.
"""
#Ya tenemos los conjuntos de atributos, ajustamos un modelo de KNN y comparamos 

#Ahora separo en conjuntos de train y test, utilizando un 20% de los datos para 
X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,
                                                    shuffle=True, stratify= y)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def KNN_3Atributos_menorVariabilidad():
    k = 5
    neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
    neigh.fit(X_train[indices_mas_chicos],y_train) #entrenamos seleccionado tres atributos
    score = neigh.score(X_test[indices_mas_chicos],y_test) #evaluamos
    print(f'Exactitud: {score}')
    

def KNN_3Atributos_mayorVariabilidad():
    k = 5
    neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
    neigh.fit(X_train[indices_mas_grandes],y_train) #entrenamos seleccionado tres atributos
    score = neigh.score(X_test[indices_mas_grandes],y_test) #evaluamos
    print(f'Exactitud: {score}')
    
def KNN_3Atributos_variabilidadMedia():
    k = 5
    neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
    neigh.fit(X_train[indices_medianos],y_train) #entrenamos seleccionado tres atributos
    score = neigh.score(X_test[indices_medianos],y_test) #evaluamos
    print(f'Exactitud: {score}')

#%%
#Ahora vamos a ir variando la cantidad de atributos tomados

n_mas_chicos = []
def get_n_mas_chicos(n):
    serieMin = varianza_pixels.copy()
    serieMin = serieMin.drop('label')  # Elimina la etiqueta 'label' y actualiza la Serie
   
    for j in range(n):                
        n_mas_chicos.append(serieMin.idxmin())    
        serieMin = serieMin.drop(serieMin.idxmin())  # Elimina el mínimo y actualiza la Serie
    return n_mas_chicos

n_mas_grandes = []
def get_n_mas_grandes(n):
    serieMax = varianza_pixels.copy()
    
    for j in range(n):
        n_mas_grandes.append(serieMax.idxmax())
        serieMax = serieMax.drop(serieMax.idxmax())  # Elimina el maximo y actualiza la Serie
    return n_mas_grandes
    
def KNN_AtributosVariables_menorVariabilidad():
    k = 5
    scores = []
    nro_atributos = []
    for n in range(1,50):
        k = 5
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_n_mas_chicos(n)
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scores.append(score)
        nro_atributos.append(n)
        
    #ahora graficamos los obtenido
    
    plt.scatter(nro_atributos, scores, label='Scores', color = 'red', s = 20)
    plt.plot(nro_atributos, scores, color='red', linestyle='--', label='Línea de Tendencia')
    plt.title('')
    plt.xlabel('Cantidad de atributos')
    plt.ylabel('Score')
    plt.show()

  
def KNN_AtributosVariables_mayorVariabilidad():
    k = 5
    scores = []
    nro_atributos = []
    for n in range(1,50):
        k = 5
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_n_mas_grandes(n)
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scores.append(score)
        nro_atributos.append(n)
        
    #ahora graficamos los obtenido
   
    plt.scatter(nro_atributos, scores, label='Scores', color = 'violet', s = 20)
    plt.plot(nro_atributos, scores, color='violet', linestyle='--', label='Línea de Tendencia')
    plt.title('')
    plt.xlabel('Cantidad de atributos')
    plt.ylabel('Score')
    plt.show()
 
