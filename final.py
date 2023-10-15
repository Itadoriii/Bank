import joblib
import pandas as pd
import numpy as np  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Carga el modelo entrenado
loaded_model = joblib.load('modelo_mlp_entrenado.pkl')

# Crea un nuevo DataFrame con datos de ejemplo
nuevos_datos = pd.DataFrame({
    'age': [35, 45, 30, 25],
    'job': ['admin.', 'blue-collar', 'technician', 'student'],
    'marital': ['married', 'single', 'married', 'single'],
    'education': ['tertiary', 'secondary', 'tertiary', 'primary'],
    'default': ['no', 'no', 'no', 'no'],
    'balance': [1500, 200, 500, 100],
    'housing': ['yes', 'no', 'yes', 'no'],
    'loan': ['no', 'no', 'no', 'no'],
    'contact': ['cellular', 'telephone', 'cellular', 'cellular'],
    'day': [10, 15, 5, 20],
    'month': ['may', 'jun', 'mar', 'nov'],
    'duration': [300, 400, 200, 100],
    'campaign': [1, 3, 2, 1],
    'pdays': [10, 20, 30, 40],
    'previous': [2, 0, 1, 0],
    'poutcome': ['success', 'unknown', 'failure', 'unknown']
})

# Crear una copia de los nuevos datos
nuevos_datos_codificados = nuevos_datos.copy()

# Crear un objeto LabelEncoder
label_encoder = LabelEncoder()

# Codificar las variables categóricas
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    nuevos_datos_codificados[col] = label_encoder.fit_transform(nuevos_datos_codificados[col])

# Realizar predicciones con el modelo cargado
predicciones = loaded_model.predict(nuevos_datos_codificados)

# Agregar las predicciones a los nuevos datos como una nueva columna
nuevos_datos_codificados['prediccion'] = predicciones

# Mostrar los resultados
print("Resultados de las predicciones:")
print(nuevos_datos_codificados[['age', 'job', 'marital', 'education', 'prediccion']])

# ...

# Cargar tus datos de prueba y realizar la evaluación final
def cargar_datos_de_prueba():
    # Cargar tus datos de prueba desde el archivo "bank-full.csv"
    bankdatafull = pd.read_csv("bank-full.csv", sep=";")

    # Copiar los datos para no modificar los originales
    X_test = bankdatafull.drop('y', axis=1).copy()  # Variables predictoras
    y_test = bankdatafull['y'].copy()  # Variable objetivo

    # Codificar las variables categóricas en X_test
    for col in categorical_columns:
        # Manejar etiquetas desconocidas con 'desconocido'
        X_test[col] = X_test[col].apply(lambda x: x if x in label_encoder.classes_ else 'unknown')
        X_test[col] = label_encoder.transform(X_test[col])

    # Convertir etiquetas reales (y_test) a valores numéricos
    y_test_numeric = y_test.map({'no': 0, 'yes': 1})

    return X_test, y_test_numeric  # Nota el cambio en y_test

# Llamado a la función cargar_datos_de_prueba
X_test, y_test = cargar_datos_de_prueba()

# Realizar la evaluación final
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Exactitud (Accuracy): {accuracy}')
print('Reporte de Clasificación:\n', report)
