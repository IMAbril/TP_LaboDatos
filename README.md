LÉEME:

Aquí encontrarás la información correspondiente para poder comprender los contenidos de 'TP02-thebeatles.zip'
En esta carpeta se encuentran dos archivos principales y una carpeta :
  * Un informe en formato PDF que detalla el trabajo realizado y las conclusiones obtenidas : TP-02-SignMNIST.pdf
  * Un archivo .py que contiene el códgio principal : sign_thebeatles.py
  * Una carpeta llamada Modulos, donde se encuentran los modulos importados en el archivo original.
     * Exploracion.py : Donde se encuentra el código correspondiente al análisis exploratorio realizado
     * KNN.py : En donde se encuentra el código correspondiente a la sección Clasificación binaria de el informe
     * KNN2.py : En donde se encuentra el código correspondiente a la sección Clasificación binaria de el informe
 Aclaración: Dada la corrección, nos surgió la duda de si debíamos evaluar los resultados de cada modelo de KNN con los datos de test como decía en el enunciado (punto 2d) o con los datos de train aplicando la técnica k-fold cross validation y usar los datos de test para la performance del modelo final. Por ello, en KNN.py se encuentra cada modelo de KNN evaluado con los datos de test y en KNN2.py se encuentra la técnica k-fold cross validation y la performance del modelo elegido. Decidimos conservar ambos archivos porque, por la escasez de tiempo, no pudimos aclarar nuestra duda como lo hicimos con la que tuvimos con el punto 3. 
     * ArbolesDecision.py : Aquí se encuentra el código correspondiente a la sección multiclase de el informe 


_______________________________________________________________________________________________________________________________________________________________________________________________
Versiones requeridas para la ejecución de archivos:
  *Name: pandas  Version: 1.5.3
  *Name: seaborn Version: 0.13.2
