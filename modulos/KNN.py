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
from sklearn.model_selection import train_test_split,cross_val_score, KFold
import matplotlib.pyplot as plt 
import random 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import seaborn as sns

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

#Vemos como se comportan las letras analizando sus componentes principales:       
def comparar_A_L():   
    X = data_L_A.drop('label', axis=1).values
    y = data_L_A['label'].values 
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Letra A', s=7, color='violet')
    plt.scatter(X_pca[y == 11, 0], X_pca[y == 11, 1], label='Letra L', s=7, color='green')
    
    plt.title('PCA: Comparación entre A y L')
    plt.xlabel('Primera Componente')
    plt.ylabel('Segunda Componente')
    plt.legend()
    plt.show()

comparar_A_L()         

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

#Ya tenemos los conjuntos de atributos, ajustamos un modelo de KNN y comparamos 

#Ahora separo en conjuntos de train y test, utilizando un 20% de los datos para 
X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,
                                                    shuffle=True, stratify= y)

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
    serieMin = varianza_pixels.copy() #hacemos una copia para 
    serieMin = serieMin.drop('label')  # Elimina la etiqueta 'label' y actualiza la Serie
   
    for j in range(n):                
        n_mas_chicos.append(serieMin.idxmin())    
        serieMin = serieMin.drop(serieMin.idxmin())  # Elimina el mínimo y actualiza la Serie
    return n_mas_chicos

n_mas_grandes = []
def get_n_mas_grandes(n):
    serieMax = varianza_pixels.copy() #hacemos una copia para no modificar el original
    
    for j in range(n):
        n_mas_grandes.append(serieMax.idxmax())
        serieMax = serieMax.drop(serieMax.idxmax())  # Elimina el maximo y actualiza la Serie
    return n_mas_grandes

n_medianos = []

def get_n_medianos(n):
    serieMed = varianza_pixels.copy() #hacemos una copia para no modificar el original
    serieMed = pd.DataFrame(serieMed)     # Convertir a DataFrame
    serieMed_ordenado = serieMed.sort_values(by='std') #ordenamos de acuerdo al 'std'
    mediana_index = len(serieMed_ordenado) // 2 #calculamos mediana
    iterador = n-1 // 2 #resto uno para no tener en cuenta la mediana
    i = 1
    mediana = serieMed_ordenado.iloc[mediana_index]['std']
    n_medianos.append( serieMed_ordenado.index[mediana_index])
   
    while i < iterador:
        # Obtener el nombre del índice anterior a la mediana
        indice_anterior = serieMed_ordenado.index[mediana_index - i] if mediana_index > 0 else None
        indices_medianos.append(indice_anterior)
        # Obtener el nombre del índice siguiente a la mediana
        indice_siguiente = serieMed_ordenado.index[mediana_index + i] if mediana_index < len(serieMed_ordenado) - 1 else None
        n_medianos.append(indice_siguiente)
        i += 1
    return n_medianos

scoresMenorVar = []
nro_atributosMenorVar = []

def KNN_AtributosVariables_menorVariabilidad():
    k = 5
    for n in range(1,21):
        k = 5
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_n_mas_chicos(n)
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scoresMenorVar.append(score)
        nro_atributosMenorVar.append(n)
        
    #ahora graficamos los obtenido
    
    # plt.scatter(nro_atributos, scores, label='Scores', color = 'red', s = 20)
    # plt.plot(nro_atributos, scores, color='red', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de la cantidad de atributos con menor variabilidad')
    # plt.xlabel('Cantidad de atributos')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()


scoresMayorVar = []
nro_atributosMayorVar = []
def KNN_AtributosVariables_mayorVariabilidad():
    k = 5
    for n in range(1,21):
        k = 5
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_n_mas_grandes(n)
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scoresMayorVar.append(score)
        nro_atributosMayorVar.append(n)
        
    #ahora graficamos los obtenido
   
    # plt.scatter(nro_atributos, scores, label='Scores', color = 'violet', s = 20)
    # plt.plot(nro_atributos, scores, color='violet', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de la cantidad de atributos con mayor variabilidad')
    # plt.xlabel('Cantidad de atributos')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()
 

scoresVarMedia = []
nro_atributosVarMedia = []
def KNN_AtributosVariables_variabilidadMedia():
    k = 5
    for n in range(1,21):
        k = 5
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_n_medianos(n)
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scoresVarMedia.append(score)
        nro_atributosVarMedia.append(n)
        
    #ahora graficamos los obtenido
   
    # plt.scatter(nro_atributos, scores, label='Scores', color = 'green', s = 20)
    # plt.plot(nro_atributos, scores, color='green', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de la cantidad de atributos con variabilidad media')
    # plt.xlabel('Cantidad de atributos')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()

def grafico_atributos_variables():
    # KNN_AtributosVariables_menorVariabilidad()
    # KNN_AtributosVariables_mayorVariabilidad() #para ejecutar este codigo sin cargar la celda completa, descomentar
    # KNN_AtributosVariables_variabilidadMedia()
    
    plt.plot(nro_atributosMenorVar, scoresMenorVar, label='Menor Variabilidad', color='red', linestyle='-')
    plt.plot(nro_atributosMayorVar, scoresMayorVar, label='Mayor Variabilidad', color='violet', linestyle='-')
    plt.plot(nro_atributosVarMedia, scoresVarMedia, label='Variabilidad Media', color='green', linestyle='-')
    
    
    plt.title('Score en función de la cantidad de atributos con diferentes niveles de variabilidad')
    plt.xlabel('Cantidad de atributos')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)  # agregamos cuadrícula para que sea más fácil visualizar
    plt.show()
grafico_atributos_variables()
#%%

#Variamos k, para conocer cual es el valor que mejor ajusta

scores_VarMed = []
k_elegido_VarMed = []

def KNN_k_variabilidadMedia():
    for k in range(1,21):
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_indices_medianos()
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scores_VarMed.append(score)
        k_elegido_VarMed.append(k)
        
    #ahora graficamos los obtenido
   
    # plt.scatter(k_elegido_VarMed, scores_VarMed, label='Scores', color = 'blue', s = 20)
    # plt.plot(k_elegido_VarMed, scores_VarMed, color='blue', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de k con 3 atributos de variabilidad media')
    # plt.xlabel('Cantidad de vecinos (k)')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()
    
scores_VarMenor = []
k_elegido_VarMenor = []

def KNN_k_variabilidadMenor():
    for k in range(1,21):
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_indices_mas_chicos()
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scores_VarMenor.append(score)
        k_elegido_VarMenor.append(k)
        
    #ahora graficamos los obtenido
   
    # plt.scatter(k_elegido_VarMenor, scores_VarMenor, label='Scores', color = 'orange', s = 20)
    # plt.plot(k_elegido_VarMenor, scores_VarMenor, color='orange', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de k con 3 atributos de menor variabilidad ')
    # plt.xlabel('Cantidad de vecinos (k)')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()

scores_VarMayor = []
k_elegido_VarMayor = []

def KNN_k_variabilidadMayor():
    for k in range(1,21):
        neigh = KNeighborsClassifier(n_neighbors= k) #iniciamos el modelo
        atributos = get_indices_mas_grandes()
        neigh.fit(X_train[atributos],y_train) #entrenamos seleccionado tres atributos
        score = neigh.score(X_test[atributos],y_test) #evaluamos
        scores_VarMayor.append(score)
        k_elegido_VarMayor.append(k)
        
    # #ahora graficamos los obtenido
   
    # plt.scatter(k_elegido_VarMayor, scores_VarMayor, label='Scores', color = 'brown', s = 20)
    # plt.plot(k_elegido_VarMayor, scores_VarMayor, color='brown', linestyle='--', label='Línea de Tendencia')
    # plt.title('Score en función de k con 3 atributos de mayor variabilidad')
    # plt.xlabel('Cantidad de vecinos (k)')
    # plt.ylabel('Score')
    # plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    # plt.show()
    
    
def grafico_k_variable():
    # KNN_k_variabilidadMayor()
    # KNN_k_variabilidadMedia() #para ejecutar este codigo sin cargar la celda completa, descomentar
    # KNN_k_variabilidadMenor()
    
    plt.plot(k_elegido_VarMed, scores_VarMed, color='blue', linestyle='--', label='Variabiliad Media')
    plt.plot(k_elegido_VarMenor, scores_VarMenor, color='orange', linestyle='--', label='Menor Variabilidad')
    plt.plot(k_elegido_VarMayor, scores_VarMayor, color='brown', linestyle='--', label='Mayor Variabilidad')
    
    plt.title('Score en funcion de la cantidad de vecinos con diferentes niveles de variabilidad')    
    plt.xlabel('Cantidad de vecinos (k)')
    plt.ylabel('Score')
    plt.legend(loc='lower center', bbox_to_anchor=(0.9, 0.5), shadow=True, ncol=1)
    plt.grid(True) #agregamos cuadricula para que sea mas facil visualizar
    plt.show()

grafico_k_variable()

#%%
#realizamos cross validation para evaluar nuestro modelo

#Para evaluar los modelos, realizamos cross validation para cada variabilidad utilizando tres atributos
#Variabilidad media

def cross_validation():
        
    # Defino un nuevo modelo KNN,con k = 5, basamos nuestra decision en lo explorado en grafico_k_variable
    knn = KNeighborsClassifier(n_neighbors=5)
    
    
    kf = KFold(n_splits= 5, shuffle=True, random_state=5)
    
    # Realiza la validación cruzada para obtener los distintos rendimientos
    
    # Utilizando atributos con menor variabilidad
    scores_menor_var = cross_val_score(knn, X_train[indices_mas_chicos], y_train, cv=kf)
    
    # Utilizando atributos con mayor variabilidad
    scores_mayor_var = cross_val_score(knn, X_train[indices_mas_grandes], y_train, cv=kf)
    
    # Utilizando atributos con variabilidad media
    scores_var_media = cross_val_score(knn, X_train[indices_medianos], y_train, cv=kf)
    
    print( 'Score con atributos de menor variabilidad: ',scores_menor_var.mean())
    print("Score con atributos de mayor variabilidad:", scores_mayor_var.mean())
    print("Score con atributos de variabilidad media:", scores_var_media.mean())

cross_validation()
