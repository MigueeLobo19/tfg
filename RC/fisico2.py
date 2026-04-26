# Limpieza de CSV eliminando valores nulos e intervalos de tiempo sin medidas
import pandas 
import numpy
import matplotlib.pyplot as plt # Añadido para poder graficar
import time

csv = pandas.read_csv('data-roomA-10T.csv', sep=';')

csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

# Leemos los datos de la sala 68
room_68 = csv[csv['room'] == 68].copy()

# Ordenamos para evitar saltos en el tiempo
room_68.sort_index(inplace=True)

# Buscamos si hay huecos de mas de 10 minutos
diferencia_t = room_68.index.to_series().diff()
huecos_68 = diferencia_t[diferencia_t > pandas.Timedelta(minutes=10)]

print(f"Análisis para la sala 68:")
print(f"Total de registros encontrados: {len(room_68)}")
print(f"Número de huecos detectados: {len(huecos_68)}")

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

# Aplicamos los multiplicadores anteriores
room_68['hvac'] = numpy.select(estado_HVAC, multiplicadores)

# Definimos el rendimiento estimado del equipo 
COP_estimado = 3.0

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / (bloqueA_rooms)

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para calcular el 1R1C
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Valores típicos de R y C
R_inicial = 0.05
C_inicial = 100000000
Asol_inicial = 2.5

col_T_int = 'V2'
col_T_ext = 'tmed'

inicio_simulacion = time.perf_counter()

def simulacion_1R1C(T_ext, Q_hvac, Rad_solar, T_int_inicial, R, C, A_sol, dt_minutos=10):
    dt_segundos = dt_minutos * 60
    n_pasos = len(T_ext)
    
    T_sim = numpy.zeros(n_pasos)
    T_sim[0] = T_int_inicial
    
    T_ext_vals = T_ext.values
    Q_hvac_vals = Q_hvac.values
    Rad_vals = Rad_solar.values # <-- Aquí van los datos del sensor (W/m2)
    
    for i in range(n_pasos - 1):
        # Flujo de calor por conducción (muros)
        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        
        # Flujo de calor por radiación (W/m2 * m2 = W)
        Q_sol = Rad_vals[i] * A_sol # <-- Multiplicamos el dato de ese instante por el tamaño de la ventana
        
        # Variación total de temperatura (la suma de todos los calores)
        dT = (flujo_paredes + Q_hvac_vals[i] + Q_sol) / C
        if Q_hvac_vals[i] != 0 and Q_hvac_vals[i] > 0:
            print(f"Paso {i} -> Q_hvac: {Q_hvac_vals[i]}")
        
        # Se aplica el método de Euler
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
        
    return T_sim




T_simulada = simulacion_1R1C(
    T_ext = datos_limpios[col_T_ext],
    Q_hvac = datos_limpios['Q_hvac'],
    Rad_solar = datos_limpios['radmed'],  # La columna con los datos del sol
    T_int_inicial = datos_limpios[col_T_int].iloc[0], 
    R = R_inicial, 
    C = C_inicial,
    A_sol = Asol_inicial  
)


# Guardamos el resultado
datos_limpios['T_simulada'] = T_simulada

fin_simulacion = time.perf_counter()
tiempo_ejecucion = fin_simulacion - inicio_simulacion

# Dibujamos las gráficas
plt.figure(figsize=(15, 7))

plt.plot(datos_limpios.index, datos_limpios[col_T_int], label='T. Interior REAL (Sensor)', color='black', linewidth=1.5)
plt.plot(datos_limpios.index, datos_limpios['T_simulada'], label='T. Interior SIMULADA (Modelo)', color='orange', linestyle='--')
plt.plot(datos_limpios.index, datos_limpios[col_T_ext], label='T. Exterior', color='blue', alpha=0.3)

plt.title(f'Modelo 1R1C - Sala 68 (R={R_inicial}, C={C_inicial})')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 1. Calculamos el error absoluto en grados Celsius
datos_limpios['error_abs'] = (datos_limpios[col_T_int] - datos_limpios['T_simulada']).abs()

# 2. Definimos qué consideramos un "error grande" (por ejemplo, más de 2 grados)
umbral_error = 20.0 

# 3. Filtramos los datos donde el error supera el umbral
fallos_modelo = datos_limpios[datos_limpios['error_abs'] > umbral_error].copy()

# 4. Mostramos los 20 errores más grandes para analizar patrones
print(f"Se han encontrado {len(fallos_modelo)} registros con un error mayor a {umbral_error}°C.")
print("\nRegistros con mayor error:")
# Ordenamos por error de mayor a menor y mostramos columnas clave
columnas_analisis = [col_T_int, 'T_simulada', 'error_abs', col_T_ext, 'hvac', 'dif_cons']
print(fallos_modelo[columnas_analisis].sort_values(by='error_abs', ascending=False))

# Variables para no repetir tanto texto
T_real = datos_limpios[col_T_int]
T_sim = datos_limpios['T_simulada']

# 1. Error Medio Absoluto (MAE - Mean Absolute Error)
mae = numpy.mean(numpy.abs(T_real - T_sim))

# 2. Error Cuadrático Medio (MSE - Mean Squared Error)
mse = numpy.mean((T_real - T_sim)**2)

# 3. Raíz del Error Cuadrático Medio (RMSE - Root Mean Squared Error)
rmse = numpy.sqrt(mse)

# R^2
ss_res = numpy.sum((T_real - T_sim)**2)          # Sum of Squares of Residuals (lo que el modelo falla)
ss_tot = numpy.sum((T_real - numpy.mean(T_real))**2) # Total Sum of Squares (la varianza natural de los datos)
r_cuadrado = 1 - (ss_res / ss_tot)

# Imprimimos los resultados con un formato limpio
print
print("\n" + "=" * 40)
print("   MÉTRICAS DE RENDIMIENTO DEL MODELO")
print("=" * 40)
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2: {r_cuadrado:.4f}")
print(f"Tiempo de ejecución         : {tiempo_ejecucion * 1000:.2f} milisegundos")
print("=" * 40)
