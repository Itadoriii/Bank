#TRABAJO INFERENCIA SECCION 412 (?)
#INTEGRANTES SEBASTIAN CASTRO, CARLOS PARADA, PABLO ZUÑIGA, DIEGO ADROVEZ.

import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt  
import seaborn as sns

#Lectura de archivos csv 
#(RECOLECCION DE DATOS)

#Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012 
bankdata = pd.read_csv("bank.csv", sep=";")
bankdatafull = pd.read_csv("bank-full.csv", sep=";")

#Created by: Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho) and Paulo Rita (ISCTE-IUL) @ 2014 
bankdataadditional = pd.read_csv("bank-additional.csv", sep=";")
bankdataadditionalfull = pd.read_csv("bank-additional-full.csv", sep=";")


print("bank full csv (SHAPE 1):")
print(bankdatafull.shape)
print("bank additional full csv(SHAPE 1):")
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
#Elimina columnas que no se usaran 
#balance = 'balance'
#bankdatafull.drop(balance, axis=1, inplace=True)
# Conteo de los niveles en las diferentes columnas categóricas
cols_cat_bankdata = ['job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month', 'poutcome', 'y']

for col in cols_cat_bankdata:
  print(f'Columna {col}: {bankdatafull[col].nunique()} subniveles')


cols_cat_bankdataadd = ['job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month','day_of_week','poutcome','y']
for col in cols_cat_bankdata:
  print(f'Columna {col}: {bankdataadditionalfull[col].nunique()} subniveles')

#Todas las columnas de tipo object tienen mas de un subnivel por lo que no eliminaremos ninguna
#Revision a las columnas tipo int64 y float64
print('Funcion describe de los dos dataframes:')
print(bankdatafull.describe())
print(bankdataadditionalfull.describe())
#Todas las columnas tienen una desviacion estandart diferente de 0 lo que significa que los datos de cada dataframe no son iguales

print("bank full csv (SHAPE 2):")
print(bankdatafull.shape)
print("bank additional full csv(SHAPE 2):")
print(bankdataadditionalfull.shape)

#= GRAFICOS PARA BANKDATAFULL 
# Lista de columnas numéricas
cols_num_bankdatafull = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
 
# Ajusta el tamaño de la figura para dar más espacio vertical
fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 40))  # Aumenta el valor de 'figsize'

# Aumenta el espacio vertical entre subplots
fig.subplots_adjust(hspace=2)  # Aumenta el valor de 'hspace'

# Genera gráficos de caja para cada variable numérica
for i, col in enumerate(cols_num_bankdatafull):
    sns.boxplot(x=bankdatafull[col], ax=ax[i])
    ax[i].set_title(col)

# Muestra los gráficos
plt.show()

# Genera gráficos de barras para cada columna categórica por separado
for col in cols_cat_bankdata:
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.countplot(x=col, data=bankdatafull)
    plt.title(f'Gráfico de barras para {col}')
    plt.xticks(rotation=45)  # Rota las etiquetas del eje x para mejorar la legibilidad
    plt.show()  # Muestra el gráfico actual y espera hasta que se cierre antes de continuar con el siguiente

#COMENTARIOS PARA LOS DATOS DE ARRIBA ... 

# Eliminar filas con "previous">100
print(f'Tamaño del set antes de eliminar registros de previous: {bankdatafull.shape}')
bankdatafull = bankdatafull[bankdatafull['previous']<=100]
print(f'Tamaño del set después de eliminar registros de previous: {bankdatafull.shape}')


#GRAFICOS PARA BANKDATAADDITIONALFULL 

# Lista de columnas numéricas
cols_num_bankdataadditionalfull = ['age', 'duration', 'campaign', 'pdays', 'previous']
cols_float_bankdataadditionalfull = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
# Ajusta el tamaño de la figura para dar más espacio vertical
fig, ax = plt.subplots(nrows=len(cols_num_bankdataadditionalfull), ncols=1, figsize=(8, 8 * len(cols_num_bankdataadditionalfull))) 

# Aumenta el espacio vertical entre subplots
fig.subplots_adjust(hspace=2)

# Genera gráficos de caja para cada variable numérica
for i, col in enumerate(cols_num_bankdataadditionalfull):
    sns.boxplot(x=bankdataadditionalfull[col], ax=ax[i])
    ax[i].set_title(col)

# Muestra los gráficos
plt.show()

# Ajusta el tamaño de la figura para dar más espacio vertical
fig, ax = plt.subplots(nrows=len(cols_float_bankdataadditionalfull), ncols=1, figsize=(8, 8 * len(cols_num_bankdataadditionalfull))) 

# Aumenta el espacio vertical entre subplots
fig.subplots_adjust(hspace=2)

# Genera gráficos de caja para cada variable numérica
for i, col in enumerate(cols_float_bankdataadditionalfull):
    sns.boxplot(x=bankdataadditionalfull[col], ax=ax[i])
    ax[i].set_title(col)

# Muestra los gráficos
plt.show()

# Genera gráficos de barras para cada columna categórica por separado
for col in cols_cat_bankdataadd:
    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.countplot(x=col, data=bankdataadditionalfull)
    plt.title(f'Gráfico de barras para {col}')
    plt.xticks(rotation=45)  # Rota las etiquetas del eje x para mejorar la legibilidad
    plt.show()  # Muestra el gráfico actual y espera hasta que se cierre antes de continuar con el siguiente


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Transformación de datos

# Codificación de variables categóricas usando LabelEncoder
label_encoder = LabelEncoder()
for col in cols_cat_bankdata:
    bankdatafull[col] = label_encoder.fit_transform(bankdatafull[col])

for col in cols_cat_bankdataadd:
    bankdataadditionalfull[col] = label_encoder.fit_transform(bankdataadditionalfull[col])

# Normalización de las columnas numéricas usando StandardScaler
scaler = StandardScaler()
bankdatafull[cols_num_bankdatafull] = scaler.fit_transform(bankdatafull[cols_num_bankdatafull])
bankdataadditionalfull[cols_num_bankdataadditionalfull] = scaler.fit_transform(bankdataadditionalfull[cols_num_bankdataadditionalfull])

# División de datos en conjuntos de entrenamiento, validación y prueba
X_full = bankdatafull.drop('y', axis=1)  # Variables predictoras
y_full = bankdatafull['y']  # Variable objetivo
X_add_full = bankdataadditionalfull.drop('y', axis=1)  # Variables predictoras
y_add_full = bankdataadditionalfull['y']  # Variable objetivo

# División de datos para bankdatafull
X_train_full, X_temp_full, y_train_full, y_temp_full = train_test_split(X_full, y_full, test_size=0.4, random_state=42)
X_val_full, X_test_full, y_val_full, y_test_full = train_test_split(X_temp_full, y_temp_full, test_size=0.5, random_state=42)

# División de datos para bankdataadditionalfull
X_train_add_full, X_temp_add_full, y_train_add_full, y_temp_add_full = train_test_split(X_add_full, y_add_full, test_size=0.4, random_state=42)
X_val_add_full, X_test_add_full, y_val_add_full, y_test_add_full = train_test_split(X_temp_add_full, y_temp_add_full, test_size=0.5, random_state=42)

# Verificación de las formas de los conjuntos de datos resultantes
print("Shapes de conjuntos de datos para bankdatafull:")
print("Entrenamiento:", X_train_full.shape, y_train_full.shape)
print("Validación:", X_val_full.shape, y_val_full.shape)
print("Prueba:", X_test_full.shape, y_test_full.shape)

print("\nShapes de conjuntos de datos para bankdataadditionalfull:")
print("Entrenamiento:", X_train_add_full.shape, y_train_add_full.shape)
print("Validación:", X_val_add_full.shape, y_val_add_full.shape)
print("Prueba:", X_test_add_full.shape, y_test_add_full.shape)

from sklearn.model_selection import train_test_split

# División de datos para bankdatafull
X_train_full, X_temp_full, y_train_full, y_temp_full = train_test_split(X_full, y_full, test_size=0.4, random_state=42)
X_val_full, X_test_full, y_val_full, y_test_full = train_test_split(X_temp_full, y_temp_full, test_size=0.5, random_state=42)

# División de datos para bankdataadditionalfull
X_train_add_full, X_temp_add_full, y_train_add_full, y_temp_add_full = train_test_split(X_add_full, y_add_full, test_size=0.4, random_state=42)
X_val_add_full, X_test_add_full, y_val_add_full, y_test_add_full = train_test_split(X_temp_add_full, y_temp_add_full, test_size=0.5, random_state=42)

# Verificación de las formas de los conjuntos de datos resultantes
print("Shapes de conjuntos de datos para bankdatafull:")
print("Entrenamiento:", X_train_full.shape, y_train_full.shape)
print("Validación:", X_val_full.shape, y_val_full.shape)
print("Prueba:", X_test_full.shape, y_test_full.shape)

print("\nShapes de conjuntos de datos para bankdataadditionalfull:")
print("Entrenamiento:", X_train_add_full.shape, y_train_add_full.shape)
print("Validación:", X_val_add_full.shape, y_val_add_full.shape)
print("Prueba:", X_test_add_full.shape, y_test_add_full.shape)
