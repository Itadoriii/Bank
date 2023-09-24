#TRABAJO INFERENCIA SECCION 412 (?)
#INTEGRANTES SEBASTIAN CASTRO, CARLOS PARADA, PABLO ZUÑIGA, DIEGO ADROVEZ.

import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt  # Agregar esta línea para importar Matplotlib
import seaborn as sns

#Lectura de archivos csv 
#(RECOLECCION DE DATOS)

#Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012 
bankdata = pd.read_csv("bank.csv", sep=";")
bankdatafull = pd.read_csv("bank-full.csv", sep=";")

#Created by: Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho) and Paulo Rita (ISCTE-IUL) @ 2014 
bankdataadditional = pd.read_csv("bank-additional.csv", sep=";")
bankdataadditionalfull = pd.read_csv("bank-additional-full.csv", sep=";")


print("bank full csv (SHAPE):")
print(bankdatafull.shape)
print("bank additional full csv(SHAPE):")
print(bankdataadditionalfull.shape)
"""
repeticion de un 10% de cada dataframe
print("bank csv:")
print(bankdata.shape) 
print("bank additional csv:")
print(bankdataadditional.shape)
"""



"""
DESCRIPCION DE LOS DATASET: 
1) Bank full csv
            "The data is related with direct marketing campaigns of a Portuguese banking institution."
            Input variables:
            # bank client data:
            1 - age (numeric)
            2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                                "blue-collar","self-employed","retired","technician","services") 
            3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
            4 - education (categorical: "unknown","secondary","primary","tertiary")
            5 - default: has credit in default? (binary: "yes","no")
            6 - balance: average yearly balance, in euros (numeric) 
            7 - housing: has housing loan? (binary: "yes","no")
            8 - loan: has personal loan? (binary: "yes","no")
            # related with the last contact of the current campaign:
            9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
            10 - day: last contact day of the month (numeric)
            11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
            12 - duration: last contact duration, in seconds (numeric)
            # other attributes:
            13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
            14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
            15 - previous: number of contacts performed before this campaign and for this client (numeric)
            16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

            Output variable (desired target):
            17 - y - has the client subscribed a term deposit? (binary: "yes","no")
            8. Missing Attribute Values: None

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
            
2)Bank Aditional full csv
            "This dataset is based on "Bank Marketing" UCI dataset"
            Input variables:
            # bank client data:
            1 - age (numeric)
            2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
            3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
            4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
            5 - default: has credit in default? (categorical: "no","yes","unknown")
            6 - housing: has housing loan? (categorical: "no","yes","unknown")
            7 - loan: has personal loan? (categorical: "no","yes","unknown")
            # related with the last contact of the current campaign:
            8 - contact: contact communication type (categorical: "cellular","telephone") 
            9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
            10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
            11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
            # other attributes:
            12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
            13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
            14 - previous: number of contacts performed before this campaign and for this client (numeric)
            15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
            # social and economic context attributes
            16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
            17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
            18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
            19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
            20 - nr.employed: number of employees - quarterly indicator (numeric)

            Output variable (desired target):
            21 - y - has the client subscribed a term deposit? (binary: "yes","no")

            
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
            8. Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label.
              These missing values can be treated as a possible class label or using deletion or imputation techniques. 

"""
#(LIMPIEZA DE DATOS)

# Verificar los valores faltantes en los DataFrame para comprobar si faltan datos. #son 0 en todos asi que no faltan datos 
#print(bankdatafull.isnull().sum())
#print(bankdataadditionalfull.isnull().sum())
                                            

#Identifica y elimina filas duplicadas
bankdatafull.drop_duplicates(inplace=True) #elimina 0 filas
bankdataadditionalfull.drop_duplicates(inplace=True) #elimina 12 filas 

print(bankdatafull.describe())

print(bankdataadditionalfull.describe())


#(EXPLORACION DE DATOS)



# 1. Resumen estadístico inicial
print(bankdatafull.describe())

# 2. Distribución de variables numéricas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(bankdatafull['age'], bins=20, kde=True)
plt.title('Distribución de Edades')
plt.subplot(1, 2, 2)
sns.boxplot(x='balance', data=bankdatafull)
plt.title('Boxplot de Saldo')
plt.show()

# 3. Distribución de variables categóricas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='job', data=bankdatafull)
plt.xticks(rotation=90)
plt.title('Distribución de Trabajos')
plt.subplot(1, 2, 2)
sns.countplot(x='marital', data=bankdatafull)
plt.title('Distribución de Estados Civiles')
plt.show()

# 4. Relaciones entre variables
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='balance', data=bankdatafull)
plt.title('Relación entre Edad y Saldo')
plt.show()

# 5. Correlaciones
# Convertir columnas categóricas en variables ficticias
bankdatafull = pd.get_dummies(bankdatafull, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'])

correlation_matrix = bankdatafull.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# 8. Análisis de valores atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(x='duration', data=bankdatafull)
plt.title('Boxplot de Duración de Llamada')
plt.show()


#(TRANSFORMACION DE DATOS)
#(DIVISION DE DATOS)
