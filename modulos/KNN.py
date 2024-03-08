# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024
"""
from exploracion import sign, cantMuestras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
#%%
#A partir del dataframe original, construimos un nuevo dataframe que contenga sólo al subconjunto de imágenes correspondientes a señas de las letras L o A.

data_L_A = sign[(sign['label'] == 0) | (sign['label'] == 11)]


#Obtuvimos la cantidad de muestras de cada letra al realizar la exploración.
cantidad_letra_A = cantMuestras[cantMuestras['letras'] == 'a']['cantidad'].values[0]
print('Cantidad de muestras de la letra A:', cantidad_letra_A)

cantidad_letra_L = cantMuestras[cantMuestras['letras'] == 'l']['cantidad'].values[0]
print('Cantidad de muestras de la letra L:', cantidad_letra_L)


diferenciaMuestral = cantidad_letra_A/cantidad_letra_L
print('Las clases están balanceadas, pues al compararlas obtenemos un valor muy cercano a uno : ', round(diferenciaMuestral,2))

#%% Ahora creamos nuestro clasificador y comenzamos a entrenarlo 

# Comenzamos separando nuestras variables 
X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label']

#Separamos en un conjunto de entrenamiento y otro de validación. Colocamos una semilla, con random-state
# Separamos en 80% entrenamiento y 20% prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

"""Ajustar un modelo de KNN considerando pocos atributos, por ejemplo
3. Probar con distintos conjuntos de 3 atributos y comparar resultados.
Analizar utilizando otras cantidades de atributos."""

#Ahora iniaciamos un modelo KNN considerando todos los atributos y un numero de K = 3, pequeño
neigh = KNeighborsClassifier(n_neighbors= 3)

#Lo entenamos con los conjuntos de X e y

neigh.fit(X,y)

#Ahora evaluamos el modelo, calculando su R^2
score = neigh.score(X, y)

print("R^2 (train ): %.2f" % score)

#######################################
#COMENTARIO PARA INFORME
#Al utilizar un numero K bajo, el modelo, se está sobreajustando


#%%

#Ahora Probamos distintos conjuntos de tres atributos (columnas) y comparamos resultados, evaluando en el conjunto de test
#para ello, listamos todas las columnas de X, ya que no contiene 'label'

lista_columnas = X.columns.tolist()

# Separamos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

#Creo dos listas, para poder realizar un grafico
iteraciones = []
scores = []
k = 5

# Probamos con distintos conjuntos de tres atributos y comparamos resultados
for i in range(5):   #Iteramos 5 veces
    columnas_Al_Azar= random.sample(lista_columnas, 3) #seleccionamos columnas al azar
    neigh = KNeighborsClassifier(n_neighbors=k) #iniciamos el modelo
    neigh.fit(X_train[columnas_Al_Azar], y_train) #lo entrenamos con las columnas seleccionadas
    score = neigh.score(X_test[columnas_Al_Azar], y_test)#lo evaluamos
    scores.append(score) 
    iteraciones.append(i)
    print(f' Score del modelo: {score}')
    
#Ahora graficamos    
plt.scatter(iteraciones, scores, label='Scores', color = 'red', s = 20)
plt.plot(iteraciones, scores, color='red', linestyle='--', label='Línea de Tendencia')
plt.title('Scores del Modelo KNN con Conjuntos de 3 Atributos')
plt.xlabel('Numero de iteración')
plt.ylabel('Score')
plt.show()
#%%

#Ahora vamos variando, no solo los atributos utilizados, sinó tambien el numero de k

#Plantamos una semilla para que la generación del random en nuestr lista_atributos, sea siempre la misma al ejecutar 
random_state = 5
# Generamos una lista de 10 numeros aleatorios entre 1 y 99
lista_atributos = [random.randint(1, 99) for _ in range(10)]
print(lista_atributos)

#Inicializamos dos variables para luego realizar graficos
atributos = []
scores = []

for k in range(1, 50):
       cant_atributos = random.choice(lista_atributos)
       columnas_Al_Azar = random.sample(lista_columnas, cant_atributos)#seleccionamos columnas al azar
       neigh = KNeighborsClassifier(n_neighbors=k) #iniciamos el modelo
       neigh.fit(X_train[columnas_Al_Azar], y_train) #lo entrenamos con las columnas seleccionadas
       score = neigh.score(X_test[columnas_Al_Azar], y_test)#lo evaluamos
       scores.append(score)
       atributos.append(cant_atributos)
       print(f'Score del modelo: {score:.2}, cantidad de vecinos: {k}, cantidad de atributos: {cant_atributos}')

#Realizamos el grafico de Score en funcion del numero de vecinos
plt.plot(range(1, 50), scores, marker='o', linestyle='-', color='violet')
plt.title('Score del Modelo en Función del Número de Vecinos')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Score')
plt.grid(True)
plt.show()

#Realizamos el grafico e score en funcion de la cantidad de atributos
plt.scatter(atributos, scores, marker='o', color='orange')
plt.title('Score del Modelo en Función de la Cantidad de Atributos')
plt.xlabel('Cantidad de Atributos')
plt.ylabel('Score')
plt.grid(True)
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#PROBAR

import numpy as np 
# Rango de valores por los que se va a mover k
valores_k = range(1, 20)
#  Cantidad de veces que vamos a repetir el experimento
Nrep = 50
# Matrices donde vamos a ir guardando los resultados
resultados_test  = np.zeros(( Nrep , len(valores_k)))
resultados_train = np.zeros(( Nrep , len(valores_k)))

# Realizamos la combinacion de todos los modelos (Nrep x k)
for i in range(Nrep):
    # Dividimos en test(30%) y train(70%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2) 
    # Generamos el modelo y lo evaluamos
    for k in valores_k:
        # Declaramos el tipo de modelo
        neigh = KNeighborsClassifier(n_neighbors = k)
        # Entrenamos el modelo (con datos de train)
        neigh.fit(X_train, Y_train) 
        # Evaluamos el modelo con datos de train y luego de test
        resultados_train[i,k-1] = neigh.score(X_train, Y_train)
        resultados_test[i,k-1]  = neigh.score(X_test , Y_test )

# Promediamos los resultados de cada repeticion
promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test  = np.mean(resultados_test , axis = 0) 

##################################################################
## Graficamos R2 en funcion de k (para train y test)
##################################################################
plt.plot(valores_k, promedios_train, label = 'Train')
plt.plot(valores_k, promedios_test, label = 'Test')
plt.legend()
plt.title('Performance del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('R^2')
plt.xticks(valores_k)
plt.ylim(0.60,1.00)
