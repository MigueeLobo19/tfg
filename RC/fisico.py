import pandas 
import numpy
import matplotlib.pyplot as plt # Añadido para poder graficar
import time

print("Comienza la simulación del modelo 1R1C...\n")

# Cargamos CSV
csv = pandas.read_csv('data-roomA-10T.csv', sep=';')
csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

# Leemos los datos de la sala 68 y ordenamos
room_68 = csv[csv['room'] == 68].copy()
room_68.sort_index(inplace=True)

# Buscamos si hay huecos de mas de 10 minutos
diferencia_t = room_68.index.to_series().diff()
huecos_68 = diferencia_t[diferencia_t > pandas.Timedelta(minutes=10)]

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

# Definimos el rendimiento o estimado del equipo 
COP_estimado = 4.5

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / (bloqueA_rooms)

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para calcular el 1R1C
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Valores típicos de R y C
R_inicial = 0.02
C_inicial = 80000000
Asol_inicial = 2

col_T_int = 'V2'
col_T_ext = 'tmed'

inicio_simulacion = time.perf_counter()
def simulacion_1R1C(T_ext, Q_hvac, Rad_solar, T_int_inicial, R, C, A_sol, dt_minutos=10):
    dt_segundos = dt_minutos * 60
    n_pasos = len(T_ext)
    
    T_sim = numpy.zeros(n_pasos)
    T_sim[0] = T_int_inicial
    
    # Extraemos los valores de las series de Pandas a arrays de Numpy (más rápido)
    T_ext_vals = T_ext.values
    Q_hvac_vals = Q_hvac.values
    Rad_vals = Rad_solar.values # <-- Aquí van los datos del sensor (W/m2)
    
    for i in range(n_pasos - 1):

        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        Q_sol = Rad_vals[i] * A_sol 
        dT = (flujo_paredes + Q_hvac_vals[i] + Q_sol) / C
        # Euler
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
        
    return T_sim

T_simulada = simulacion_1R1C(
    T_ext = datos_limpios[col_T_ext],
    Q_hvac = datos_limpios['Q_hvac'],
    Rad_solar = datos_limpios['radmed'],  
    T_int_inicial = datos_limpios[col_T_int].iloc[0], 
    R = R_inicial, 
    C = C_inicial,
    A_sol = Asol_inicial  
)

# Guardamos el resultado
datos_limpios['T_simulada'] = T_simulada

fin_simulacion = time.perf_counter()
tiempo_ejecucion = fin_simulacion - inicio_simulacion

# Calculo de error absoluto
datos_limpios['error_abs'] = (datos_limpios[col_T_int] - datos_limpios['T_simulada']).abs()

fin_simulacion = time.perf_counter()
tiempo_ejecucion = fin_simulacion - inicio_simulacion

#umbral_error = 10.0 
#fallos_modelo = datos_limpios[datos_limpios['error_abs'] > umbral_error].copy()
#print(f"Se han encontrado {len(fallos_modelo)} registros con un error mayor a {umbral_error}°C.")
#print("\nRegistros con mayor error:")
# Ordenamos por error de mayor a menor y mostramos columnas clave
#columnas_analisis = [col_T_int, 'T_simulada', 'error_abs', col_T_ext, 'hvac', 'dif_cons']
#print(fallos_modelo[columnas_analisis].sort_values(by='error_abs', ascending=False))

T_real = datos_limpios[col_T_int]
T_sim = datos_limpios['T_simulada']

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
print(f"Tiempo de ejecución         : {tiempo_ejecucion:.2f} segundos")

# Dibujamos las gráficas
plt.figure(figsize=(15, 7))

plt.plot(datos_limpios.index, datos_limpios[col_T_int], label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(datos_limpios.index, datos_limpios['T_simulada'], label='Temperatura interior simulada', color='orange', linestyle='--')
plt.plot(datos_limpios.index, datos_limpios[col_T_ext], label='Temperatura exterior', color='blue', alpha=0.3)

plt.title(f'Modelo 1R1C: R={R_inicial}, C={C_inicial})')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
