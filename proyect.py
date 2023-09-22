#TRABAJO INFERENCIA SECCION 412 (?)
#INTEGRANTES SEBASTIAN CASTRO, CARLOS PARADA, PABLO ZUÑIGA, DIEGO ADROVEZ.
import pandas as pd 

#Lectura de archivos csv 

bankdatafull = pd.read_csv("bank-full.csv", sep=";")
bankdata = pd.read_csv("bank.csv", sep=";")
bankdataadditional = pd.read_csv("bank-additional.csv", sep=";")
bankdataadditionalfull = pd.read_csv("bank-additional-full.csv", sep=";")


#print(bankdata.to_string())  #print de todo el dataframe 
print("bank full csv:")
print(bankdatafull.shape) 
print("bank csv:")
print(bankdata.shape) 
print("bank additional csv:")
print(bankdataadditional.shape) 
print("bank additional full csv:")
print(bankdataadditionalfull.shape) 

bankdataadditionalfull.info()
"""
#bankdatafull.info()
(45211, 17)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 17 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   age        45211 non-null  int64  
 1   job        45211 non-null  object
 2   marital    45211 non-null  object
 3   education  45211 non-null  object
 4   default    45211 non-null  object
 5   balance    45211 non-null  int64
 6   housing    45211 non-null  object
 7   loan       45211 non-null  object
 8   contact    45211 non-null  object
 9   day        45211 non-null  int64
 10  month      45211 non-null  object
 11  duration   45211 non-null  int64
 12  campaign   45211 non-null  int64
 13  pdays      45211 non-null  int64
 14  previous   45211 non-null  int64
 15  poutcome   45211 non-null  object
 16  y          45211 non-null  object
dtypes: int64(7), object(10)


#bankdataadditionalfull.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   age             41188 non-null  int64
 1   job             41188 non-null  object
 2   marital         41188 non-null  object
 3   education       41188 non-null  object
 4   default         41188 non-null  object
 5   housing         41188 non-null  object
 6   loan            41188 non-null  object
 7   contact         41188 non-null  object
 8   month           41188 non-null  object
 9   day_of_week     41188 non-null  object
 10  duration        41188 non-null  int64
 11  campaign        41188 non-null  int64
 12  pdays           41188 non-null  int64
 13  previous        41188 non-null  int64
 14  poutcome        41188 non-null  object
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object
dtypes: float64(5), int64(5), object(11)

"""

