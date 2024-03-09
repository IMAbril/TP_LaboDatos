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


#Como queremos probar con distintos conjuntos de tres atributos, listamos todos los atributos

lista_columnas = X.columns.tolist()

#Vamos a analizar con un numero de vecinos determinado
k = 5

#Inicializo dos variables para luego poder graficar
scores = []
modelo = []
atributos_elegidos = []

#ACÁ VARIAMOS LOS CONJUNTOS DE ATRIBUTOS
# Probamos con distintos conjuntos de tres atributos y comparamos resultados
for i in range(5):   #Iteramos 5 veces
    columnas_Al_Azar= random.sample(lista_columnas,3) #seleccionamos columnas al azar
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
#AHORA VAMOS A VARIAR LA CANTIDAD DE ATRIBUTOS

# Probamos con distintos conjuntos de tres atributos y comparamos resultados

# Supongamos que ya tienes X, Y y lista_columnas definidos
X = data_L_A.drop('label', axis=1)  # Conservamos todas las columnas excepto 'label'
Y = data_L_A['label']

# Dividimos en test(30%) y train(70%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Rango de valores por los que se va a mover k
valores_k = range(1, 10)

# AHORA VAMOS A VARIAR LA CANTIDAD DE ATRIBUTOS
lista_columnas = X.columns.tolist()

# Inicializamos dos variables para luego realizar gráficos
atributos = []
scores = []

for cant_atributos in range(1, 9):  # Probamos con atributos desde 1 a 5
    scores_atributos = []  # Guardamos los scores por cantidad de atributos
    for i in range(5):  # Iteramos 5 veces por conjunto de atributos
        random.seed(5)
        columnas_Al_Azar = random.sample(lista_columnas, cant_atributos)
        neigh = KNeighborsClassifier(n_neighbors=5)  # Iniciamos el modelo con k=5
        neigh.fit(X_train[columnas_Al_Azar], y_train)  # Entrenamos con las columnas seleccionadas
        score = neigh.score(X_test[columnas_Al_Azar], y_test)  # Evaluamos el modelo
        scores_atributos.append(score)
    prom_score = np.mean(scores_atributos) #Calculamos el promedio de los score por atributos
    scores.append(prom_score) #guardamos info para grafico
    atributos.append(cant_atributos)
    print(f'Promedio de score para {cant_atributos} atributos: {prom_score:.2f}')

# Graficamos el rendimiento en función de la cantidad de atributos
plt.plot(atributos, scores, marker='o', linestyle='-', color='blue')
plt.title('Score del Modelo en Función de la Cantidad de Atributos')

#%%

#AHORA VAMOR A VARIAR LOS VALORES DE K
X = data_L_A.drop('label', axis=1)  # Conservamos todas las columnas excepto 'label'
Y = data_L_A['label']

valores_k = range(1, 9)
scores_k = []

# Iteramos sobre diferentes valores de k
for k in valores_k:
    random.seed(6)
    columnas_Al_Azar = random.sample(lista_columnas, 3)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train[columnas_Al_Azar], y_train)
    score = neigh.score(X_test[columnas_Al_Azar], y_test)
    scores_k.append(score)
    print(f'Score del modelo con k={k}: {score:.2}, Atributos Elegidos: {columnas_Al_Azar}')

# Graficamos el rendimiento en función de k
plt.plot(valores_k, scores_k, marker='o', linestyle='-', color='green')
plt.title('Score del Modelo en Función de k, con 3 atributos')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Score')
plt.grid(True)
plt.show()


plt.xlabel('Cantidad de Atributos')
plt.ylabel('Score Promedio')
plt.grid(True)
plt.show()
