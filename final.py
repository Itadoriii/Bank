import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk, scrolledtext

# Carga el modelo entrenado
loaded_model = joblib.load('modelo_mlp_entrenado.pkl')

# Crear una ventana principal
window = tk.Tk()
window.title("Resultados de Predicciones y Estadísticas")
window.geometry("800x600")

# Crear un control de pestañas para mostrar las pestañas de Predicciones y Estadísticas
tab_control = ttk.Notebook(window)

# Crear una pestaña para Predicciones
tab_predicciones = ttk.Frame(tab_control)
tab_control.add(tab_predicciones, text="Predicciones")

# Crea un nuevo DataFrame con datos de ejemplo
bankdata = pd.read_csv("bank-full.csv", sep=";")
bankdata = bankdata.drop('y', axis=1)

# Definir una función para codificar variables categóricas
def codificar_variables_categoricas(data, encoder, columns):
    data_encoded = data.copy()
    for col in columns:
        data_encoded[col] = encoder.fit_transform(data_encoded[col])
    return data_encoded

# Variables categóricas en el dataset
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Crear y ajustar un LabelEncoder
label_encoder = LabelEncoder()
bankdata_encoded = codificar_variables_categoricas(bankdata, label_encoder, categorical_columns)

# Realizar predicciones con el modelo cargado
predicciones = loaded_model.predict(bankdata_encoded)
bankdata['prediccion'] = predicciones

# Crear un cuadro de texto desplazable para mostrar los resultados de las predicciones
text_area_predicciones = scrolledtext.ScrolledText(tab_predicciones, wrap=tk.WORD, width=60, height=20)
text_area_predicciones.insert(tk.INSERT, "Resultados de las predicciones:\n")
text_area_predicciones.insert(tk.INSERT, bankdata[['age', 'job', 'marital', 'education', 'prediccion']].to_string(index=False))
text_area_predicciones.grid(row=0, column=0, sticky="nsew")
tab_predicciones.grid_rowconfigure(0, weight=1)
tab_predicciones.grid_columnconfigure(0, weight=1)

# Crear una pestaña para Estadísticas
tab_estadisticas = ttk.Frame(tab_control)
tab_control.add(tab_estadisticas, text="Estadísticas")

# Cargar tus datos de prueba y realizar la evaluación final
def cargar_datos_de_prueba():
    # Cargar tus datos de prueba desde el archivo "bank-full.csv"
    bankdatafull = pd.read_csv("bank-full.csv", sep=";")
    X_test = bankdatafull.drop('y', axis=1).copy()  # Variables predictoras
    y_test = bankdatafull['y'].copy()  # Variable objetivo

    for col in categorical_columns:
        X_test[col] = X_test[col].apply(lambda x: x if x in label_encoder.classes_ else 'unknown')
        X_test[col] = label_encoder.transform(X_test[col])

    y_test_numeric = y_test.map({'no': 0, 'yes': 1})
    return X_test, y_test_numeric

# Llamado a la función cargar_datos_de_prueba
X_test, y_test = cargar_datos_de_prueba()

# Realizar la evaluación final
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Crear un cuadro de texto desplazable para mostrar las estadísticas
text_area_estadisticas = scrolledtext.ScrolledText(tab_estadisticas, wrap=tk.WORD, width=60, height=10)
text_area_estadisticas.insert(tk.INSERT, f'Exactitud (Accuracy): {accuracy}\n')
text_area_estadisticas.insert(tk.INSERT, 'Reporte de Clasificación:\n')
text_area_estadisticas.insert(tk.INSERT, report)
text_area_estadisticas.grid(row=0, column=0, sticky="nsew")
tab_estadisticas.grid_rowconfigure(0, weight=1)
tab_estadisticas.grid_columnconfigure(0, weight=1)

# Añadir las pestañas al control de pestañas
tab_control.pack(fill="both", expand=True)

# Iniciar la ventana principal
window.mainloop()
