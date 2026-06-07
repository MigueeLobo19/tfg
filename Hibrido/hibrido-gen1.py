import json
import sys

import pandas 
import numpy
import matplotlib.pyplot as plt 
import psutil
import tracemalloc
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time

if len(sys.argv) < 2:
    print("❌ Error: Tienes que pasar el archivo de configuración.")
    print("Uso: python mi_script.py config.json")
    sys.exit(1)

ruta_config = sys.argv[1]

print(f"Cargando configuración desde: {ruta_config}")
with open(ruta_config, 'r') as archivo:
    config = json.load(archivo)

dataset = config['dataset']
sala_seleccionada = config['room']
COP_estimado = config['COP_estimado']
R_inicial = config['R_inicial']
C_inicial = config['C_inicial']
Asol_inicial = config['Asol_inicial']
salas = config['salas']
HVAC_off = config['HVAC_off']
HVAC_calor = config['HVAC_calor']
HVAC_frio = config['HVAC_frio']
t_int = config['t_int']
t_ext = config['t_ext']
radmed = config['radmed']
dif_cons = config['dif_cons']
fecha = config['fecha']


print("Comienza la simulación del modelo híbrido y la evaluación de la generalización...\n")

# Cargamos CSV
csv = pandas.read_csv(dataset, sep=';')
csv.columns = csv.columns.str.strip()
csv[fecha] = pandas.to_datetime(csv[fecha], utc=True)
csv.set_index(fecha, inplace=True)

# Leemos los datos de la sala 68 y ordenamos
room_68 = csv[csv[salas] == sala_seleccionada].copy()
room_68.sort_index(inplace=True)

# Salas en bloque A
bloqueA_rooms = csv[salas].nunique()

# Estado del HVAC
estado_HVAC = [room_68[HVAC_off] == 1, room_68[HVAC_calor] == 1, room_68[HVAC_frio] == 1]

# En función de si el aporte calorífico es positivo (calefacción) o negativo (aire acondicionado)
multiplicadores = [
    0,   
    1,   
    -1   
]

room_68['hvac'] = numpy.select(estado_HVAC, multiplicadores)

# Definimos el rendimiento estimado del equipo 
COP_estimado = 4.5

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para predecir con svr
datos_limpios = room_68.dropna(subset=[t_int, t_ext, 'Q_hvac', radmed]).copy()

# Comienza el modelo físico 
print("Ejecutando simulación física 1R1C...")
R_fija = 0.02
C_fija = 80000000
Asol_fijo = 2.0

col_T_int = t_int
col_T_ext = t_ext   

def simulacion_1R1C(T_ext, Q_hvac, Rad_solar, T_int_inicial, R, C, A_sol, dt_minutos=10):
    dt_segundos = dt_minutos * 60
    n_pasos = len(T_ext)
    
    T_sim = numpy.zeros(n_pasos)
    T_sim[0] = T_int_inicial
    
    T_ext_vals = T_ext.values
    Q_hvac_vals = Q_hvac.values
    Rad_vals = Rad_solar.values 
    
    for i in range(n_pasos - 1):
        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        Q_sol = Rad_vals[i] * A_sol 
        dT = (flujo_paredes + Q_hvac_vals[i] + Q_sol) / C
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
        
    return T_sim

inicio_simulacion = time.perf_counter()
datos_limpios['T_simulada'] = simulacion_1R1C(
    T_ext = datos_limpios[col_T_ext],
    Q_hvac = datos_limpios['Q_hvac'],
    Rad_solar = datos_limpios['radmed'],
    T_int_inicial = datos_limpios[col_T_int].iloc[0], 
    R = R_fija, 
    C = C_fija,
    A_sol = Asol_fijo
)

print("Entrenando modelo para corregir los errores del modelo 1R1C...")
inicio_entrenamiento = time.perf_counter()

# Calculamos el error
datos_limpios['residuo_fisico'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']

# Hacemos que el modelo conozca el comportamiento, es decir, aprenda en función de la hora y dia de la semana
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek
datos_limpios['mes'] = datos_limpios.index.month 

# Entradas y salidas del modelo
X = datos_limpios[[t_ext, 'hvac', 'hora', 'dia_semana', radmed, 'T_simulada']]
y = datos_limpios['residuo_fisico']

print("\nFiltrando por estaciones...")
# Entrenamiento: Verano (Junio, julio y agosto)
filtro_verano = datos_limpios['mes'].isin([6, 7, 8])
X_train = X[filtro_verano]
y_train = y[filtro_verano]

# Test: Invierno (Diciembre, enero y febrero)
filtro_invierno = datos_limpios['mes'].isin([12, 1, 2])  
X_test = X[filtro_invierno]
y_test = y[filtro_invierno]

# Escalado para que las magnitudes se estandaricen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SVR
modelo_svr = SVR(kernel='rbf', C=10, epsilon=0.1)

print("\nEntrenando modelo SVR para detectar error")
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

# Predicción sobre datos test (Invierno)
inicio_prediccion = time.perf_counter()
# Tomamos valor incial del consumo de RAM
proceso_test = psutil.Process(os.getpid())
ram_antes_test_mb = proceso_test.memory_info().rss / (1024 * 1024)
psutil.cpu_percent(interval=None)
tracemalloc.start()
y_pred_test = modelo_svr.predict(X_test_scaled)
fin_prediccion = time.perf_counter()
memoria_actual_test, pico_maximo_test = tracemalloc.get_traced_memory()
tracemalloc.stop() 
cpu_usada_test = psutil.cpu_percent(interval=None)

# CPU y RAM al terminar
ram_despues_train_mb = proceso_train.memory_info().rss / (1024 * 1024)

pico_maximo_train_mb = pico_maximo_train / (1024 * 1024)
pico_maximo_test_mb = pico_maximo_test / (1024 * 1024)
tiempo_pred = fin_prediccion - inicio_prediccion

# Temperatura del modelo híbrido
T_real_test = datos_limpios.loc[y_test.index, col_T_int] 
T_fisica_test = X_test['T_simulada']                    
T_hib_test = T_fisica_test + y_pred_test 
residuos_test = T_real_test - T_hib_test                

# Cálculo métricas
mae = numpy.mean(numpy.abs(T_real_test - T_hib_test))
mse = numpy.mean((T_real_test - T_hib_test)**2)
rmse = numpy.sqrt(mse)
ss_res = numpy.sum((T_real_test - T_hib_test)**2)         
ss_tot = numpy.sum((T_real_test - numpy.mean(T_real_test))**2) 
r2 = 1 - (ss_res / ss_tot)

print("RESULTADOS DEL MODELO HÍBRIDO (Test en Invierno)")
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
plt.plot(eje_x_seguido, T_real_test, label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(eje_x_seguido, T_hib_test, label='Temperatura modelo híbrido', color='orange', linestyle='--')

plt.title('Modelo Híbrido con 1R1C + SVR')
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
plt.title('Error del Modelo Híbrido para evaluar generalización')
plt.ylabel('Error en Grados (°C)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(posiciones_etiquetas, fechas_etiquetas, rotation=45)
plt.show()
