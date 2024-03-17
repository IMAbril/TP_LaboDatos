#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°2
laboratorio de Datos - Verano 2024
Autores : Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del grupo: The Beatles 
Fecha : Marzo 2024

KNN versión 2: Obtención de las métricas para comparar modelos a partir de k-fold cross validation
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

carpeta = './datasets/'
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
#Separamos los datos en conjunto de variable de entrada y variable resultado 
X = data_L_A.drop('label', axis=1)  #Conservamos todas las columnas excepto 'label'
y = data_L_A['label'] #Variable resultado

#Separo los datos en conjuntos de train y test (o dev y eval), utilizando un 20% de los datos para test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,
                                                    shuffle=True, stratify= y)


#%%
#Para probar con distintos conjuntos de n atributos, teniendo en cuenta la variabilidad pixel a pixel entre las muestras de las letras A y L
#Elegimos considerar aquellos con desviación estándar mayor, menor e intermedia 

informacion_df = data_L_A.describe()

std_pixels = informacion_df.loc['std'].drop('label').sort_values() 

def get_3conjuntosNpixeles_variabilidad(n):
    serieStd = std_pixels.copy()
    #Indices píxeles con variabilidad más alta
    indices_mas_grandes = list(serieStd.iloc[-n:].index.values)
    #Indices píxeles con variabilidad más pequeña
    indices_mas_chicos = list(serieStd.iloc[0:n].index.values)
    #Indices píxeles con variabilidad intermedia
    indice_central = len(serieStd) // 2 # Calculamos indice central. En caso de haber 2, tomamos el primero.
    indices_medianos = list(serieStd.iloc[indice_central-(n//2):indice_central+(n//2)+(n%2)].index.values) #Tomamos los pixeles en un radio de n//2 respecto al del central, incluído este último
    return {'intermedia':indices_medianos, 'mayor':indices_mas_grandes, 'menor':indices_mas_chicos}

#%%

def KNN_Natributos_distintavariabilidad_kfijo(k, n):
    conjuntos = get_3conjuntosNpixeles_variabilidad(n)
    modelos = pd.DataFrame()
    for variabilidad, pixeles in conjuntos.items():
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train[pixeles], y_train, cv=5) #Entrena y valida el modelo, 
        mean_score = np.mean(scores)
        modelo_data = {'variabilidad':variabilidad,'knn':knn,'mean_score':mean_score, 'atributos':pixeles, 'cantAtributos':n}
        modelo_df = pd.DataFrame([modelo_data])
        modelos = pd.concat([modelos, modelo_df], ignore_index=True)
    return modelos        

#Ajustamos un modelo de KNN para cada conjunto de 3 atributos y comparamos 
modelos_3atributos = KNN_Natributos_distintavariabilidad_kfijo(5,3) #De acá se obtiene que el mejor modelo es el de variabilidad intermedia
print("-------------Métricas obtenidas con validación cruzada de modelos de 3 atributos distintos------------------ ")
print(modelos_3atributos[['variabilidad','mean_score']])

#%%
#Ahora vamos a ir variando la cantidad de atributos tomados de 1 a 20 

def KNN_NAtributos_distintaVariabilidad_kfijo(k):
    modelos_porCantAtributos = pd.DataFrame()
    for n in range(1,21):
        modelos = KNN_Natributos_distintavariabilidad_kfijo(k,n)
        modelos_porCantAtributos = pd.concat([modelos_porCantAtributos, modelos], ignore_index=True)
    return modelos_porCantAtributos

modelos_porCantAtributos = KNN_NAtributos_distintaVariabilidad_kfijo(5)

def grafico_atributos_variables():
    # Filtramos el DataFrame para obtener los datos de cada serie que vamos a graficar
    var_intermedia = modelos_porCantAtributos[modelos_porCantAtributos['variabilidad'] == 'intermedia']
    var_mayor = modelos_porCantAtributos[modelos_porCantAtributos['variabilidad'] == 'mayor']
    var_menor = modelos_porCantAtributos[modelos_porCantAtributos['variabilidad'] == 'menor']
    
    # Ajustamos el tamaño del gráfico
    plt.figure(figsize=(10, 6))
   
    plt.plot(var_intermedia['cantAtributos'], var_intermedia['mean_score'], label='Variabilidad Intermedio')
    plt.plot(var_mayor['cantAtributos'], var_mayor['mean_score'], label='Variabilidad Mayor')
    plt.plot(var_menor['cantAtributos'], var_menor['mean_score'], label='Variabilidad Menor')
    
    
    plt.title('Score en función de la cantidad de atributos con diferentes niveles de variabilidad')
    plt.xlabel('Cantidad de atributos')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)  # agregamos cuadrícula para que sea más fácil visualizar
    plt.show()
grafico_atributos_variables()    

#%%
#Ahora variamos el k de 1 a 20, con cantidad de atributos fija igual a 3

def KNN_20Kvariable():
    modelos_porK = pd.DataFrame()
    for k in range(1, 21):
        modelos = KNN_Natributos_distintavariabilidad_kfijo(k, 3)
        modelos['k']= k
        modelos_porK = pd.concat([modelos_porK, modelos], ignore_index=True)
    return modelos_porK

modelos_porK = KNN_20Kvariable()

def grafico_k_variable():    
    # Filtramos el DataFrame para obtener los datos de cada serie que vamos a graficar
    var_intermedia = modelos_porK[modelos_porK['variabilidad'] == 'intermedia']
    var_mayor = modelos_porK[modelos_porK['variabilidad'] == 'mayor']
    var_menor = modelos_porK[modelos_porK['variabilidad'] == 'menor']
    plt.figure(figsize=(10, 6))
   
    plt.plot(var_intermedia['k'], var_intermedia['mean_score'], label='Variabilidad Intermedio')
    plt.plot(var_mayor['k'], var_mayor['mean_score'], label='Variabilidad Mayor')
    plt.plot(var_menor['k'], var_menor['mean_score'], label='Variabilidad Menor')
    
    plt.title('Score del modelo en función de k y variabilidd de atributos')
    plt.xlabel('Cantidad de Vecinos (k)')
    plt.ylabel('Score')
    plt.legend()  
    plt.grid(True)  # colocamos grilla 
    plt.show()

grafico_k_variable()



#%%
#Ahora calculamos la performance del modelo
#Con base a lo analizado y representado en los graficos, tomamos como hiperparametros k = 7 y cant de atributos 3, con variabilidad intermedia
def performance_mejor_modelo():
    mejor_modelo = KNeighborsClassifier(n_neighbors=7)
    atributos = get_3conjuntosNpixeles_variabilidad(3)['intermedia']  # Extraer solo los nombres de columnas
    mejor_modelo.fit(X_train[atributos], y_train)
    
    y_pred = mejor_modelo.predict(X_test[atributos])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejor modelo: {mejor_modelo}, Exactitud:{accuracy} ")
    
    # Calculamos el informe de clasificación en el conjunto de eval o test
    classification_rep_test = classification_report(y_test, y_pred)
    
    # Imprimimos las métricas del mejor modelo para saber todos sus datos
    print("Informe de clasificación en conjunto de test:")
    print(classification_rep_test)
    
    conf_mat_test = confusion_matrix(y_test, y_pred)
    
    # Creamos un DataFrame para la matriz de confusión con etiquetas personalizadas
    conf_mat_df = pd.DataFrame(conf_mat_test, index=['Letra A','Letra L'], columns=['Letra A', 'Letra L'])
    
    # Creamos un heatmap de la matriz de confusión
    sns.heatmap(conf_mat_df, annot=True, fmt='d', cmap='viridis', cbar=False, linewidths=0.5, linecolor='black')
    
    # Añadimos etiquetas y título
    plt.title('Matriz de Confusión del Mejor Modelo')
    plt.xlabel('Predicción')
    plt.ylabel('Valores reales')
    
    plt.show()

performance_mejor_modelo()


    
