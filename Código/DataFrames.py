# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Nombre del grupo: The Beatles 
Fecha : Febrero 2024
"""
import pandas as pd
import Normalizar as n
from inline_sql import sql, sql_val

#Creamos DataFrames vacios correspondientes al Modelo Relacional de nuestro DER

columnas_sede = ['id_sede', 'Nombre', 'Tipo', 'codigo_pais','URl_Red_Social']
sedes_df = pd.DataFrame(columns=columnas_sede)


columnas_pais = ['Codigo','PBI', 'Nombre','Region', 'Nivel_Ingreso']
paises_df = pd.DataFrame(columns = columnas_pais)

 
columnas_seccion = ['id_sede','Descripcion']
secciones_df = pd.DataFrame(columns=columnas_seccion)


columnas_redSocial=['Nombre','Url']
redSocial_df=pd.DataFrame(columns=columnas_redSocial)

#Ahora importamos los datos ya limpios:
    
carpeta = "/Users/Roju2/OneDrive/Desktop/Código/tablas limpias/"

sedes = pd.read_csv(carpeta+'sedes.csv')
PBI =pd.read_csv(carpeta + 'pbi.csv')
regiones =pd.read_csv(carpeta + 'regiones.csv')
secciones =pd.read_csv(carpeta + 'secciones.csv')


sedes_df['Nombre'] = sedes['sede_desc_ingles']
sedes_df['id_sede']=sedes['sede_id']
sedes_df['Tipo']=sedes['sede_tipo']
sedes_df['codigo_pais']=sedes['pais_iso_3']


paises_df['Codigo'] = PBI['Country Code']
paises_df['PBI'] = PBI['2022 PBI']
paises_df['Nombre'] = regiones['TableName']
paises_df['Region'] = regiones['Region']
paises_df['Nivel_Ingreso'] = regiones['IncomeGroup']

secciones_df['id_sede'] = secciones['sede_id']
secciones_df['Descripcion'] = secciones['sede_desc_castellano']

#Para importar los datos de la red social, hay que extraerlos. 

redSocial_df['id_sede'] = sedes['sede_id']
redSocial_df['Url'] = sedes['redes_sociales']

#Como los atributos eran compuestos, llevamos a primera forma normal con una funcion de nuestro modulo.

redSocial_df = n.acomodar(redSocial_df)
#Para eliminar dependencias funcionales transitivas y llevar a 3FN, utilizamos consultas SQL para separar dependencias parciales y representar nuestro Modelo relacional
sedes_df = sql^"""
                    SELECT  s.Nombre, s.id_sede, s.Tipo, s.codigo_pais, rs.Url as URl_Red_Social
                    FROM sedes_df as s
                    INNER JOIN
                    redSocial_df as rs
                    ON s.id_sede = rs.id_sede

 """
redSocial_df = sql^"""
                    SELECT Url, Nombre
                    FROM redSocial_df
"""





