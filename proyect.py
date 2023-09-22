#TRABAJO INFERENCIA SECCION 412 (?)
#INTEGRANTES SEBASTIAN CASTRO, CARLOS PARADA, PABLO ZUÑIGA, DIEGO ADROVEZ.
import pandas as pd 

bankdata = pd.read_csv("bank-full.csv")
#print(bankdata.to_string())  #print de todo el dataframe 
print(bankdata.shape) #muestra las filas y las columnas existentes en el conjunto de datos, en este caso arroja (45211, 1), lo que es un error
#print(bankdata)
#bankdata.info()

