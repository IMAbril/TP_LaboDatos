# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Nombre del Grupo: The Beatles 
Fecha : Febrero 2024
"""

import pandas as pd

#Dominio del atributo Nombre de la relación redSocial_df

redes = ["facebook","instagram","youtube","twitter","linkedin","flickr"]

def separar(texto):
    """
    Separa un string en una lista de elementos utilizando '  //  ' como delimitador.
    """
    if texto is not None:
        texto_componentes = texto.strip(' //').split('  //  ')
        return texto_componentes
    else:
        return []
    

def obtenerNombreRed(Url):
    for red in redes: 
        if red in Url:
            return red 
        
def acomodar(df):
    columnas=['id_sede','Nombre','Url']
    res =pd.DataFrame(columns = columnas)
    res.columns =columnas
    for i in range(1,len(df)):
        if pd.notnull(df.iloc[i]['Url']) :
            urls =separar(df.iloc[i]['Url'])
            for j in range(0,len(urls)):
                url = urls[j]
                if obtenerNombreRed(url) != None:
                    nombre_red = obtenerNombreRed(url)
                    res = pd.concat([res, pd.DataFrame({'id_sede': [df.iloc[i]['id_sede']], 'Nombre': [nombre_red], 'Url': [url]})], ignore_index=True)     

    return res 

