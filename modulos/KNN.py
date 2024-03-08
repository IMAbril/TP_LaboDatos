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


k = 5

# Probamos con distintos conjuntos de tres atributos y comparamos resultados
for _ in range(5):   #Iteramos 5 veces
    columnas_Al_Azar= random.sample(lista_columnas, 3) #seleccionamos columnas al azar
    neigh = KNeighborsClassifier(n_neighbors=k) #iniciamos el modelo
    neigh.fit(X_train[columnas_Al_Azar], y_train) #lo entrenamos con las columnas seleccionadas
    score = neigh.score(X_test[columnas_Al_Azar], y_test)#lo evaluamos
    print(f' Score del modelo: {score}')
    
