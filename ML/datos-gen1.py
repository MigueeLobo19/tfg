import pandas 
import numpy
import matplotlib.pyplot as plt 
import time
import psutil
import tracemalloc
import os

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

# Aprende en verano-invierno
print("\nFiltrando por estaciones...")
# Junio, julio y agosto entrena
filtro_verano = datos_limpios['mes'].isin([6, 7, 8])
# Diciembre, enero y febrero test
filtro_invierno = datos_limpios['mes'].isin([12, 1, 2])  

X_train = X[filtro_verano]
y_train = y[filtro_verano]

X_test = X[filtro_invierno]
y_test = y[filtro_invierno]

#Escalado para que las magnitudes se estandaricen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SVR
modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1) 

print("\nEntrenando en verano")
# Tomamos valor incial del consumo de RAM
proceso_train = psutil.Process(os.getpid())
ram_antes_train_mb = proceso_train.memory_info().rss / (1024 * 1024)

# Vaciamos el cache de CPU para medir el uso real durante el entrenamiento
psutil.cpu_percent(interval=None)
# Activamos el seguimiento de memoria con tracemalloc
tracemalloc.start()

inicio_entrenamiento = time.perf_counter()
modelo_svr.fit(X_train_scaled, y_train)
fin_entrenamiento = time.perf_counter()
# Tomamos valor final del consumo de RAM
memoria_actual_train, pico_maximo_train = tracemalloc.get_traced_memory()
tracemalloc.stop() 
tiempo_train = fin_entrenamiento - inicio_entrenamiento
cpu_usada_train = psutil.cpu_percent(interval=None)

# Predicción en invierno
print("Predicción en invierno")
# Tomamos valor incial del consumo de RAM
proceso_test = psutil.Process(os.getpid())
ram_antes_test_mb = proceso_test.memory_info().rss / (1024 * 1024)
psutil.cpu_percent(interval=None)
tracemalloc.start()
inicio_prediccion = time.perf_counter()
y_pred_test = modelo_svr.predict(X_test_scaled)
fin_prediccion = time.perf_counter()
tiempo_pred = fin_prediccion - inicio_prediccion

memoria_actual_test, pico_maximo_test = tracemalloc.get_traced_memory()
tracemalloc.stop() 
cpu_usada_test = psutil.cpu_percent(interval=None)

# CPU y RAM al terminar
ram_despues_train_mb = proceso_train.memory_info().rss / (1024 * 1024)

pico_maximo_train_mb = pico_maximo_train / (1024 * 1024)
pico_maximo_test_mb = pico_maximo_test / (1024 * 1024)

residuos_test = y_test - y_pred_test

# Cálculo métricas
mae = numpy.mean(numpy.abs(y_test - y_pred_test))
mse = numpy.mean((y_test - y_pred_test)**2)
rmse = numpy.sqrt(mse)
ss_res = numpy.sum((y_test - y_pred_test)**2)         
ss_tot = numpy.sum((y_test - numpy.mean(y_test))**2) 
r2 = 1 - (ss_res / ss_tot)

print("RESULTADOS SVR DESPUES DEL ENTRENAMIENTO")
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2   (Coef. de Determinación): {r2:.4f}\n")
      
print("COSTE COMPUTACIONAL")
print(f"Tiempo Entrenamiento : {tiempo_train:.4f} segundos")
print(f"Tiempo Predicción    : {tiempo_pred:.2f} segundos")
print(f"Uso de CPU durante entrenamiento: {cpu_usada_train}%")
print(f"Uso de CPU durante test: {cpu_usada_test}%")
print(f"Pico máximo de RAM usado durante entrenamiento: {pico_maximo_train_mb:.2f} MB")
print(f"Pico máximo de RAM usado durante test: {pico_maximo_test_mb:.2f} MB")

# Gráfica
plt.figure(figsize=(15, 7))
eje_x_seguido = numpy.arange(len(y_test))

plt.plot(eje_x_seguido, y_test, label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(eje_x_seguido, y_pred_test, label='Temperatura predicha con SVR', color='orange', linestyle='--')

plt.title('Modelo Datos con SVR para evaluar generalización')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
posiciones_etiquetas = numpy.linspace(0, len(y_test) - 1, 10, dtype=int)
fechas_etiquetas = [y_test.index[i].strftime('%d-%b') for i in posiciones_etiquetas] 
plt.xticks(posiciones_etiquetas, fechas_etiquetas, rotation=45)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(eje_x_seguido, residuos_test, label='Error en la simulación', color='crimson', linewidth=1)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
plt.fill_between(eje_x_seguido, residuos_test, 0, 
                 where=(residuos_test >= 0), color='crimson', alpha=0.3)
plt.fill_between(eje_x_seguido, residuos_test, 0, 
                 where=(residuos_test < 0), color='blue', alpha=0.3)
plt.title('Error del Modelo de Datos SVR para evaluar generalización')
plt.ylabel('Error en Grados (°C)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(posiciones_etiquetas, fechas_etiquetas, rotation=45)
plt.show()
