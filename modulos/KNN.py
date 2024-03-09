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
import pandas as pd



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

"""
Ajustar un modelo KNN considerando pocos atributos, por ejemplo 3.
Probar con distintos conjuntos de 3 atributos y comparar resultados.
Analizar utilizando otras cantidades de atributos 
"""

#Como queremos probar con distintos conjuntos de tres atributos, listamos todos los atributos

lista_columnas = X.columns.tolist()

#Vamos a analizar con un numero de vecinos determinado
k = 5

#Inicializo dos variables para luego poder graficar
scores = []
modelo = []
atributos_elegidos = []
# Probamos con distintos conjuntos de tres atributos y comparamos resultados
for i in range(5):   #Iteramos 5 veces
    columnas_Al_Azar= random.sample(lista_columnas, 3)
    random.state = 5 #seleccionamos columnas al azar
    neigh = KNeighborsClassifier(n_neighbors=k) #iniciamos el modelo
    neigh.fit(X_train[columnas_Al_Azar], y_train) #lo entrenamos con las columnas seleccionadas
    score = neigh.score(X_test[columnas_Al_Azar], y_test)#lo evaluamos
    scores.append(score)
    modelo.append(i)
    atributos_elegidos.append(columnas_Al_Azar)
    print(f'Modelo {i} = Score: {score:.2}, Atributos Elegidos: {columnas_Al_Azar}')

#Creamos y guardamos una tabla con la informacion
tabla = [['Modelo', 'Score', 'Atributos Elegidos']]

for i in range(5):
    tabla.append([modelo[i], f'{scores[i]:.2f}', atributos_elegidos[i]])

# Guardar la tabla como una imagen
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=tabla, colLabels=None, cellLoc='center', loc='center')

#plt.savefig('ScoreAtributosAzar.png')
plt.show()

plt.scatter(modelo, scores, label='Scores', color = 'red', s = 20)
plt.plot(modelo, scores, color='red', linestyle='--', label='Línea de Tendencia')
plt.title('Scores del Modelo KNN con Conjuntos de 3 Atributos')
plt.xlabel('Modelo')
plt.ylabel('Score')
plt.show()

#%%
"""
Comparar modelos de KNN utilizando distintos atributos y distintos
valores de k (vecinos). Para el análisis de los resultados, tener en
cuenta las medidas de evaluación (por ejemplo, la exactitud) y la
cantidad de atributos.
"""

#Ahora vamos a comparar distintos modelos variando atributos y valores de K

#Declaramos nuevamente las variables
X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label']

random_state = 5
# Generar una lista de 10 numeros aleatorios entre 1 y 99
lista_atributos = [random.randint(1, 99) for _ in range(10)]

print(lista_atributos)

#Inicializamos dos variables para luego realizar graficos
atributos = []
scores = []

for k in range(1, 50):
        for n in lista_atributos:
           cant_atributos = n
           columnas_Al_Azar = random.sample(lista_columnas, cant_atributos)#seleccionamos columnas al azar
           neigh = KNeighborsClassifier(n_neighbors=k) #iniciamos el modelo
           neigh.fit(X_train[columnas_Al_Azar], y_train) #lo entrenamos con las columnas seleccionadas
           score = neigh.score(X_test[columnas_Al_Azar], y_test)#lo evaluamos
           scores.append(score)
           atributos.append(cant_atributos)
           print(f'Score del modelo: {score:.2}, cantidad de vecinos: {k}, cantidad de atributos: {cant_atributos}')


# Realizamos el gráfico de Score en función del número de vecinos
plt.plot(range(1, 50), scores[:49], marker='o', linestyle='-', color='violet')
plt.title('Score del Modelo en Función del Número de Vecinos')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# Realizamos el gráfico de score en función de la cantidad de atributos
plt.scatter(atributos, scores, marker='o', color='orange')
plt.title('Score del Modelo en Función de la Cantidad de Atributos')
plt.xlabel('Cantidad de Atributos')
plt.ylabel('Score')
plt.grid(True)
plt.show()

