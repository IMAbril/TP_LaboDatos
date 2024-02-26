# -*- coding: utf-8 -*-
"""
Trabajo Práctico N°1
laboratorio de Datos - Verano 2024
Autores :Ibarra, Abril; Vassolo, Francisco; Dominguez,Rocio
Nombre del Grupo: The Beatles 
Fecha : Febrero 2024
"""

import DataFrames as tp
import consultasSQL as cn
from inline_sql import sql, sql_val
import matplotlib.pyplot as plt 
from matplotlib import ticker   # Para agregar separador de miles
from matplotlib import rcParams # Para modificar el tipo de letra
import pandas as pd
import numpy as np
import seaborn as sns

#%% i) Cantidad de sedes por región geográfica. Mostrarlos de manera decreciente por dicha cantidad.

#Consulta para extraer región geográfica por sede

sedes = tp.sedes_df
paises = tp.paises_df
cantSedesPorRegion = sql ^ """ 
                        SELECT COUNT(sedes.id_sede) AS cantidad_sedes, paises.Region AS region
                        FROM sedes
                        INNER JOIN paises 
                        ON sedes.codigo_pais = paises.Codigo
                        GROUP BY paises.region
                        ORDER BY cantidad_sedes                        
                    """
fig, ax = plt.subplots()
ax.barh(cantSedesPorRegion['region'],cantSedesPorRegion['cantidad_sedes'])

ax.set_title('Cantidad de sedes por región geográfica', fontsize='large')
ax.set_xlabel('Cantidad de sedes', fontsize='medium')              

#%% ii) Boxplot, por cada región geográfica, del PBI per cápita 2022 de los países donde Argentina tiene una delegación.
# Mostrar todos los boxplots en una misma figura, ordenados por la mediana de cada región.         
paises = tp.paises_df
pbi_region = sql ^ """
                        SELECT CAST(REPLACE(PBI, ',', '.') AS DECIMAL(9,2)) AS PBI, Region
                        FROM paises
                        ORDER BY Region, PBI
                   """
                   

pbi_region = sql ^ """
                        SELECT PBI, Region
                        FROM pbi_region
                        ORDER BY Region, PBI
                   """

region_mediana = sql ^ """
                            SELECT Region, MEDIAN(PBI) AS mediana
                            FROM pbi_region
                            GROUP BY Region
                            ORDER BY mediana
                        """
pbi_region_mediana = sql ^ """
                                SELECT pr.Region, pr.PBI, rm.mediana 
                                FROM pbi_region AS pr
                                INNER JOIN region_mediana AS rm
                                ON pr.Region = rm.Region
                                ORDER BY mediana
                            """
pbi_region_mediana_zoom = sql ^ """
                                SELECT *
                                FROM pbi_region_mediana
                                WHERE Region == 'Sub-Saharan Africa' OR Region == 'South Asia'
                                """

fig, ax = plt.subplots()
    
rcParams['font.family'] = 'sans-serif'           # Modifica el tipo de letra
rcParams['axes.spines.right']  = False            # Elimina linea derecha   del recuadro
rcParams['axes.spines.left']   = True             # Agrega  linea izquierda del recuadro
rcParams['axes.spines.top']    = False            # Elimina linea superior  del recuadro
rcParams['axes.spines.bottom'] = False            # Elimina linea inferior  del recuadro
sns.boxplot(data=pbi_region_mediana, x='Region', y='PBI')
# Agrega titulo, etiquetas al eje Y 
ax.set_title('PBI 2022 por región geográfica ')
ax.set_xlabel('')
ax.set_ylabel('PBI per capita (U$S)')
plt.yticks(np.arange(0, 120001, 10000))  # Ajuste para incrementos de 10,000 en el eje Y
ax.set_ylim(0,120000)
plt.xticks(np.arange(7),labels=['SSA','SA','MENA','LAC','EAP','ECA','NA'])
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}")); 

#Boxplot con zoom

fig, ax = plt.subplots()
    
rcParams['font.family'] = 'sans-serif'           # Modifica el tipo de letra
rcParams['axes.spines.right']  = False            # Elimina linea derecha   del recuadro
rcParams['axes.spines.left']   = True             # Agrega  linea izquierda del recuadro
rcParams['axes.spines.top']    = False            # Elimina linea superior  del recuadro
rcParams['axes.spines.bottom'] = False            # Elimina linea inferior  del recuadro
sns.boxplot(data=pbi_region_mediana_zoom, x='Region', y='PBI')
# Agrega titulo, etiquetas al eje Y 
ax.set_title('PBI 2022 por región geográfica ')
ax.set_xlabel('')
ax.set_ylabel('PBI per capita (U$S)')
plt.yticks(np.arange(0, 10000, 1000))  # Ajuste para incrementos de 1,000 en el eje Y
ax.set_ylim(0,8000)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}")); 

#%%

# Extraer los datos de Sedes y PBI
SedesPBI = cn.tablaFinal_df

SedesPBI_df = sql ^ """
                        SELECT Sedes, "PBI per Cápita 2022 (U$S)"
                        FROM SedesPBI                     
                        """    
fig, ax = plt.subplots()

rcParams['font.family'] = 'sans-serif'           
rcParams['axes.spines.right'] = False            
rcParams['axes.spines.left']  = True             
rcParams['axes.spines.top']   = False          

width = 10000                      # Cada esta cantidad de dolares
                                                 
bins = np.arange(1,120000, width)             
counts, bins = np.histogram(SedesPBI_df['PBI per Cápita 2022 (U$S)'], bins=bins)      

center = (bins[:-1] + bins[1:]) / 2                                   

ax.bar(x=center,          
           height=counts,       
           width=width,         
           align='center',      
           color=['darkseagreen'],    
           edgecolor='black')   
ax.set_title('PBI por Cantidad de Sedes')
ax.set_xlabel('PBI per Cápita 2022 (U$S) cada 10 mil')
ax.set_ylabel('Cantidad de sedes')
ax.set_ylim(0,30)
bin_edges = [max(0, i-1) for i in bins]              

#creamos una función para reducir los miles, para volver de esta forma, el grafico más legible.Fuentes consultadas, en el informe
def reducirmiles(n):
    if n >= 1e6:
        return f'{n/1e6:.0f}M'
    elif n >= 1e3:
        return f'{n/1e3:.0f}'
    else:
        return str(n)


labels = [f'({reducirmiles(int(edge))},{reducirmiles(int(bin_edges[i+1]))})' 
          for i, edge in enumerate(bin_edges[:-1])]

       
ax.set_xticks(bin_edges[:-1])                       
ax.set_xticklabels(labels, rotation=45, fontsize=12) 
ax.tick_params(bottom = False)                      
