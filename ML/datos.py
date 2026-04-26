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

# Cargar datos del dataset
csv = pandas.read_csv('data-roomA-10T.csv', sep=';')
csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

room_68 = csv[csv['room'] == 68].copy()
room_68.sort_index(inplace=True)

bloqueA_rooms = csv['room'].nunique()
estado_HVAC = [room_68['V5_0'] == 1, room_68['V5_1'] == 1, room_68['V5_2'] == 1]
room_68['hvac'] = numpy.select(estado_HVAC, [0, 1, -1])

COP_estimado = 3.0
#room_68['dif_cons_limpio'] = room_68['dif_cons'].clip(upper=25.0)
#room_68['dif_cons_suavizado'] = room_68['dif_cons_limpio'].rolling(window=3, min_periods=1).mean()
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']
room_68['Q_hvac'] = room_68['Q_hvac'].clip(lower=-3500.0, upper=3500.0)

# Cargamos los datos necesarios
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Hacemos que el modelo conozca el comportamiento, es decir, aprenda en función de la hora y dia de la semana
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek 

# Entradas y salidas del sistema
columnas_X = ['tmed', 'hvac', 'hora', 'dia_semana', 'radmed']
X = datos_limpios[columnas_X]
y = datos_limpios['V2'] 

#Dividir datos para test (30%) y entrenamiento (70%)
# Se usa suffle=false para que los datos esten en orden
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

print(f"Total de registros limpios : {len(datos_limpios)}")
print(f"Registros de Entrenamiento : {len(X_train)}")
print(f"Registros de Prueba (Test) : {len(X_test)}")

#Escalado para que las magnitudes se estandaricen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# modelo SVR: con kernerl rbf para capturar la inercia térmica
modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1) 

print("\nEntrenando modelo SVR")
inicio_entrenamiento = time.perf_counter()

modelo_svr.fit(X_train_scaled, y_train)

fin_entrenamiento = time.perf_counter()
tiempo_train = fin_entrenamiento - inicio_entrenamiento

# Predicción sobre datos test
inicio_prediccion = time.perf_counter()

y_pred_test = modelo_svr.predict(X_test_scaled)

fin_prediccion = time.perf_counter()
tiempo_pred = fin_prediccion - inicio_prediccion

# Cálculo métricas
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = numpy.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("RESULTADOS SVR DESPUES DEL ENTRENAMIENTO\n")
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2   (Coef. de Determinación): {r2:.4f}\n")
      
print("COSTE COMPUTACIONAL\n")
print(f"Tiempo Entrenamiento : {tiempo_train:.4f} segundos")
print(f"Tiempo Predicción    : {tiempo_pred * 1000:.2f} milisegundos")

# --- 9. GRÁFICA COMPARATIVA ---
plt.figure(figsize=(15, 7))

# Graficamos SOLO el segmento de Test para ver cómo se comporta en "el mundo real"
plt.plot(y_test.index, y_test, label='T. Interior REAL (Sensor)', color='black', linewidth=1.5)
plt.plot(y_test.index, y_pred_test, label='T. Predicha (SVR Black-Box)', color='magenta', linestyle='--')
plt.plot(y_test.index, X_test['tmed'], label='T. Exterior', color='blue', alpha=0.3)

plt.title('Evaluación SVR')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

