# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :
Fecha : Febrero 2024
"""
import pandas as pd

carpeta = "carpeta correspondiente"
#Creo DataFrame vacio
sedes = pd.read_csv(carpeta+'sedes.csv')

columnas_sede = ['id_sede', 'Nombre', 'Tipo', 'Estado', 'codigo_pais','Redes_Sociales']
sedes_df = pd.DataFrame(columns=columnas_sede)

#Cargo DataSet
sedes_df['Nombre'] = sedes['sede_desc_ingles']
sedes_df['id_sede']=sedes['sede_id']
sedes_df['Tipo']=sedes['sede_tipo']
#sedes_df['Estado']=sedes[''] #la sacamos ?
sedes_df['codigo_pais']=sedes['pais_iso_3']
sedes_df['Redes_Sociales']=sedes['redes_sociales']



PBI =pd.read_csv(carpeta + 'pbi.csv')
regiones =pd.read_csv(carpeta + 'regiones.csve')

columnas_pais = ['Codigo','PBI', 'Nombre','Region', 'Nivel-Ingreso']
paises_df = pd.DataFrame(columns = columnas_pais)

paises_df['Codigo'] = PBI['Country Code']
paises_df['PBI'] = PBI['2022 PBI']
paises_df['Nombre'] = regiones['TableName']
paises_df['Region'] = regiones['Region']
paises_df['Nivel_Ingreso'] = regiones['IncomeGroup']


columnas_seccion = ['id_sede','Descripcion']
secciones_df = pd.DataFrame(columns=columnas_seccion)

secciones =pd.read_csv(carpeta + 'secciones.csv')

secciones_df['id_sede'] = secciones['sede_id']
secciones_df['Descripcion'] = secciones['sede_desc_castellano']



