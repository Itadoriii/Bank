import pandas as pd

# Cargar el archivo CSV generado con las predicciones
df = pd.read_csv('predicciones.csv')
df.info()

# Lista de columnas categóricas
columnas_categoricas = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','prediccion']

# Codificación one-hot para las columnas categóricas
df_encoded = pd.get_dummies(df, columns=columnas_categoricas)

# Calcular la media y la moda para todas las columnas
media_columnas = df_encoded.mean()
desviacion_es = df_encoded.std()

# Imprimir la media y la desviación estándar
print("Media de las predicciones:")
print(media_columnas.to_string())

print("\nDesviación estándar de las predicciones:")
print(desviacion_es.to_string())
''' 
Supongamos que queremos probar si la edad promedio es significativamente diferente de 40 años:
H0 : μ = 40
H1 : μ != 40

'''
from scipy.stats import ttest_1samp

# Nivel de significancia
alpha = 0.05

# Valor esperado de la media
mu = 40

# Realizar la prueba de hipótesis
statistic, p_value = ttest_1samp(df['age'], mu)

# Comparar el valor p con el nivel de significancia
if p_value < alpha:
    print(f"Rechazamos la hipótesis nula. Hay evidencia suficiente para afirmar que la edad promedio es diferente de {mu}.")
else:
    print(f"No podemos rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que la edad promedio es diferente de {mu}.")
