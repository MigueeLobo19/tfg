import json
import string

import pandas 
import numpy
import matplotlib.pyplot as plt
import time
import psutil
import tracemalloc
import os
import sys


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

print("Comienza la simulación del modelo 1R1C...\n")

# Cargamos CSV
csv = pandas.read_csv(dataset, sep=';')
csv.columns = csv.columns.str.strip()
csv[fecha] = pandas.to_datetime(csv[fecha], utc=True)
csv.set_index(fecha, inplace=True)

# Leemos los datos de la sala 68 y ordenamos
room_68 = csv[csv[salas] == sala_seleccionada].copy()
room_68.sort_index(inplace=True)

# Buscamos si hay huecos de mas de 10 minutos
#diferencia_t = room_68.index.to_series().diff()
#huecos_68 = diferencia_t[diferencia_t > pandas.Timedelta(minutes=10)]

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
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / (bloqueA_rooms)

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para modelar el 1R1C
datos_limpios = room_68.dropna(subset=[t_int, t_ext, 'Q_hvac', radmed]).copy()

# Valores típicos de R y C
# R_inicial = 0.02
# C_inicial = 80000000
# Asol_inicial = 2

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
        # Euler
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
        
    return T_sim

inicio_simulacion = time.perf_counter()
# Tomamos valor incial del consumo de RAM
proceso = psutil.Process(os.getpid())
ram_antes_mb = proceso.memory_info().rss / (1024 * 1024)

# Vaciamos el cache de CPU para medir el uso real durante el entrenamiento
psutil.cpu_percent(interval=None)
# Activamos el seguimiento de memoria con tracemalloc
tracemalloc.start()
T_simulada = simulacion_1R1C(
    T_ext = datos_limpios[col_T_ext],
    Q_hvac = datos_limpios['Q_hvac'],
    Rad_solar = datos_limpios['radmed'],  
    T_int_inicial = datos_limpios[col_T_int].iloc[0], 
    R = R_inicial, 
    C = C_inicial,
    A_sol = Asol_inicial  
)
# Tomamos valor final del consumo de RAM
fin_simulacion = time.perf_counter()
memoria_actual, pico_maximo = tracemalloc.get_traced_memory()
tracemalloc.stop() 
cpu_usada = psutil.cpu_percent(interval=None)
tiempo_ejecucion = fin_simulacion - inicio_simulacion

# Guardamos el resultado
datos_limpios['T_simulada'] = T_simulada

# Calculo de error absoluto
datos_limpios['error_abs'] = (datos_limpios[col_T_int] - datos_limpios['T_simulada']).abs()


T_real = datos_limpios[col_T_int]
T_sim = datos_limpios['T_simulada']
datos_limpios['residuo'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']

# Error Medio Absoluto
mae = numpy.mean(numpy.abs(T_real - T_sim))
# Error Cuadrático Medio
mse = numpy.mean((T_real - T_sim)**2)
# Raíz del Error Cuadrático Medio 
rmse = numpy.sqrt(mse)
# R^2
ss_res = numpy.sum((T_real - T_sim)**2)         
ss_tot = numpy.sum((T_real - numpy.mean(T_real))**2) 
r_cuadrado = 1 - (ss_res / ss_tot)

# Imprimimos resultados
print("MÉTRICAS DE RENDIMIENTO: ")
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2: {r_cuadrado:.4f}")

print("COSTE COMPUTACIONAL")
print(f"Tiempo de ejecución         : {tiempo_ejecucion:.2f} segundos")
print(f"Uso de CPU durante simulación: {cpu_usada}%")
print(f"Pico máximo de RAM usado durante simulación: {pico_maximo:.2f} MB")

# Dibujamos las gráficas
plt.figure(figsize=(15, 7))

plt.plot(datos_limpios.index, datos_limpios[col_T_int], label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(datos_limpios.index, datos_limpios['T_simulada'], label='Temperatura interior simulada', color='orange', linestyle='--')

plt.title(f'Modelo físico: 1R1C')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(datos_limpios.index, datos_limpios['residuo'], label='Error en la simulación', color='crimson', linewidth=1)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
plt.fill_between(datos_limpios.index, datos_limpios['residuo'], 0, 
                 where=(datos_limpios['residuo'] >= 0), color='crimson', alpha=0.3)
plt.fill_between(datos_limpios.index, datos_limpios['residuo'], 0, 
                 where=(datos_limpios['residuo'] < 0), color='blue', alpha=0.3)
plt.title('Error del Modelo Físico 1R1C')
plt.ylabel('Error en Grados (°C)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()