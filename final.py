import joblib
import pandas as pd

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

from sklearn.preprocessing import LabelEncoder

# Crear una copia de los nuevos datos
nuevos_datos_codificados = nuevos_datos.copy()

# Crear un objeto LabelEncoder
label_encoder = LabelEncoder()

# Codificar las variables categóricas
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
    nuevos_datos_codificados[col] = label_encoder.fit_transform(nuevos_datos_codificados[col])

# Realizar predicciones con el modelo cargado
predicciones = loaded_model.predict(nuevos_datos_codificados)

# Agregar las predicciones a los nuevos datos como una nueva columna
nuevos_datos_codificados['prediccion'] = predicciones

# Mostrar los resultados
print("Resultados de las predicciones:")
print(nuevos_datos_codificados[['age', 'job', 'marital', 'education', 'prediccion']])
