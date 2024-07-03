import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Cargando los datos
file_path = "C:/Users/ben19/Downloads/Articulo/cigarrillos.xlsx"
df = pd.read_excel(file_path)

# Mostrar los nombres de las columnas para verificar
print("Columnas originales del archivo Excel:", df.columns)

# Convertir las columnas a tipo numérico si es necesario
df['ipc'] = pd.to_numeric(df['ipc'], errors='coerce')
df['poblacion'] = pd.to_numeric(df['poblacion'], errors='coerce')
df['paquetes'] = pd.to_numeric(df['paquetes'], errors='coerce')
df['ingreso'] = pd.to_numeric(df['ingreso'], errors='coerce')
df['impuesto'] = pd.to_numeric(df['impuesto'], errors='coerce')
df['precio'] = pd.to_numeric(df['precio'], errors='coerce')

# Mostrar información general del dataset
print(df.info())

# Generando el modelo de regresión lineal múltiple con las variables relevantes
X = df[['ipc', 'poblacion', 'paquetes', 'ingreso', 'impuesto']]
y = df['precio']
X = sm.add_constant(X)  # Agregar término constante

modelo_reducido = sm.OLS(y, X).fit()

# Mostrar resumen del nuevo modelo
print("\nResumen del modelo reducido:")
print(modelo_reducido.summary())

# Obtener los valores de R, R^2 y R^2 ajustada del nuevo modelo
r_reducido = np.sqrt(modelo_reducido.rsquared)
r2_reducido = modelo_reducido.rsquared
r2_adj_reducido = modelo_reducido.rsquared_adj

# Mostrar los valores en consola
print("\nValores de R, R^2 y R^2 ajustada del modelo reducido:")
print(f"R: {r_reducido}")
print(f"R^2: {r2_reducido}")
print(f"R^2 ajustada: {r2_adj_reducido}")

# Intervalo de confianza del nuevo modelo
print("\nIntervalo de confianza del modelo reducido:")
print(modelo_reducido.conf_int())

# Distribución normal de los residuos del nuevo modelo (Shapiro-Wilk)
residuos_reducido = modelo_reducido.resid
shapiro_test_reducido = stats.shapiro(residuos_reducido)
print("\nDistribución normal de los residuos (Shapiro-Wilk) del modelo reducido:")
print("Estadístico de prueba:", shapiro_test_reducido[0])
print("Valor p:", shapiro_test_reducido[1])

# Homocedasticidad del nuevo modelo (Prueba de Breusch-Pagan)
bp_test_reducido = sm.stats.diagnostic.het_breuschpagan(residuos_reducido, modelo_reducido.model.exog)
print("\nHomocedasticidad (Prueba de Breusch-Pagan) del modelo reducido:")
print("Estadístico de prueba:", bp_test_reducido[0])
print("Valor p:", bp_test_reducido[1])

# Matriz de correlación entre las variables relevantes
variables_interes = df[['ipc', 'poblacion', 'paquetes', 'ingreso', 'impuesto']]
matriz_correlacion = variables_interes.corr()
print("\nMatriz de correlación entre las variables:")
print(matriz_correlacion)

# Análisis de Inflación de Varianza (VIF)
print("\nAnálisis de Inflación de Varianza (VIF):")
vif_data_reducido = pd.DataFrame()
vif_data_reducido["Predictor"] = X.columns
vif_data_reducido["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data_reducido)

# Autocorrelación del nuevo modelo (Durbin-Watson)
durbin_watson_reducido = sm.stats.stattools.durbin_watson(residuos_reducido)
print("\nAutocorrelación (Durbin-Watson) del modelo reducido:", durbin_watson_reducido)

# Identificación de posibles valores atípicos o influyentes
# Atipicidad (Gráficamente)
sns.set(style="whitegrid")
residuos_df_reducido = pd.DataFrame({'Fitted': modelo_reducido.fittedvalues, 'Residuos': residuos_reducido})
sns.residplot(x='Fitted', y='Residuos', data=residuos_df_reducido, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1})
plt.axhline(y=0, color='gray', linestyle='--', lw=2)
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos para Identificar Valores Atípicos o Influyentes")
plt.show()

# Predicciones del nuevo modelo
predicciones_reducido = modelo_reducido.predict(X)
df['predicciones_reducido'] = predicciones_reducido

# Gráfico de predicciones vs. observaciones del nuevo modelo
plt.figure(figsize=(10, 6))
plt.scatter(df['precio'], df['predicciones_reducido'], alpha=0.5, label='Predicciones Reducido')
plt.plot(df['precio'], df['precio'], color='red', linestyle='--', lw=2, label='Línea de 45°')
plt.xlabel('Precio observado')
plt.ylabel('Precio predicho')
plt.title('Precio observado vs. predicho (Modelo Reducido)')
plt.legend()
plt.show()

# Gráficos de Dispersión de Precio vs. Variables Predictoras del nuevo modelo
plt.figure(figsize=(12, 6))
for i, col in enumerate(['ipc', 'poblacion', 'paquetes', 'ingreso', 'impuesto'], start=1):
    plt.subplot(2, 3, i)
    sns.regplot(x=col, y='precio', data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'lw': 1})
    plt.title(f'Precio vs. {col}')
plt.suptitle('Gráficos de Regresión de Precio vs. Variables Predictoras (Modelo Reducido)', y=1.02)
plt.tight_layout()
plt.show()

# Visualización de la Matriz de Correlación
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables')
plt.show()
