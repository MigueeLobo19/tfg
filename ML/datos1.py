import pandas 
import numpy
import matplotlib.pyplot as plt 
import time

# Liberías sklearn para machine learning
# Librería para dividir datos para entrenamiento y test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importa el algoritmo de regresión SVR
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Comienza la simulación del modelo ML y la evaluación de la generalización...\n")

# Cargamos CSV
csv = pandas.read_csv('data-roomA-10T.csv', sep=';')
csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

# Leemos los datos de la sala 68 y ordenamos
room_68 = csv[csv['room'] == 68].copy()
room_68.sort_index(inplace=True)

# Salas en bloque A
bloqueA_rooms = csv['room'].nunique()

# Estado del HVAC
estado_HVAC = [room_68['V5_0'] == 1, room_68['V5_1'] == 1, room_68['V5_2'] == 1]

# En función de si el aporte calorífico es positivo (calefacción) o negativo (aire acondicionado)
multiplicadores = [
    0,   
    1,   
    -1   
]

room_68['hvac'] = numpy.select(estado_HVAC, multiplicadores)

# Definimos el rendimiento estimado del equipo 
COP_estimado = 3.0

#room_68['dif_cons_limpio'] = room_68['dif_cons'].clip(upper=25.0)
#room_68['dif_cons_suavizado'] = room_68['dif_cons_limpio'].rolling(window=3, min_periods=1).mean()

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para 
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Hacemos que el modelo conozca el comportamiento, es decir, aprenda en función de la hora y dia de la semana
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek 
datos_limpios['mes'] = datos_limpios.index.month 

# Entradas y salidas del modelo
columnas_X = ['tmed', 'hvac', 'hora', 'dia_semana', 'radmed']
X = datos_limpios[columnas_X]
y = datos_limpios['V2'] 

# Aprende en primavera-verano
print("\nFiltrando por estaciones...")
# Abril y mayo entrena
filtro_primavera = datos_limpios['mes'].isin([1, 2])
# Enero y febrero test
filtro_invierno = datos_limpios['mes'].isin([8, 9])  

X_train = X[filtro_primavera]
y_train = y[filtro_primavera]

X_test = X[filtro_invierno]
y_test = y[filtro_invierno]

#Escalado para que las magnitudes se estandaricen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SVR
modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1) 

print("\nEntrenando en primavera-verano")
inicio_entrenamiento = time.perf_counter()
modelo_svr.fit(X_train_scaled, y_train)
fin_entrenamiento = time.perf_counter()
tiempo_train = fin_entrenamiento - inicio_entrenamiento

# Predicción en invierno
print("Predicción en invierno")
inicio_prediccion = time.perf_counter()
y_pred_test = modelo_svr.predict(X_test_scaled)
fin_prediccion = time.perf_counter()
tiempo_pred = fin_prediccion - inicio_prediccion

# Cálculo métricas
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = numpy.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("RESULTADOS SVR DESPUES DEL ENTRENAMIENTO")
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2   (Coef. de Determinación): {r2:.4f}\n")
      
print("COSTE COMPUTACIONAL")
print(f"Tiempo Entrenamiento : {tiempo_train:.4f} segundos")
print(f"Tiempo Predicción    : {tiempo_pred:.2f} segundos")

# Gráfica
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(y_test.index, y_pred_test, label='Temperatura predecida con ML', color='orange', linestyle='--')
plt.plot(y_test.index, X_test['tmed'], label='Temperatura exterior', color='blue', alpha=0.3)

plt.title('Modelo Datos con SVR para evaluar generalización')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

