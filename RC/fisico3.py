import pandas 
import numpy
import matplotlib.pyplot as plt 
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
COP_estimado = 4.5

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / (bloqueA_rooms)

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# ==============================================================================
# 🚨 CAMBIO 1: DETECTOR DE LATIDO E IMPUTACIÓN DE TELEMETRÍA 🚨
# ==============================================================================
# Identificamos el fallo del sensor porque el consumo base (standby de ~0.23) 
# cae a exactamente 0.0.
filtro_sensor_muerto = room_68['dif_cons'] == 0.0

# Extraemos la potencia histórica cuando la máquina SÍ funciona correctamente
datos_sanos_calor = room_68[(room_68['hvac'] == 1) & (room_68['Q_hvac'] > 0)]
datos_sanos_frio = room_68[(room_68['hvac'] == -1) & (room_68['Q_hvac'] < 0)]

potencia_media_calor = datos_sanos_calor['Q_hvac'].mean() if not datos_sanos_calor.empty else 2000
potencia_media_frio = datos_sanos_frio['Q_hvac'].mean() if not datos_sanos_frio.empty else -2000

# Definimos el horario lectivo (Lunes a Viernes, de 08:00 a 20:00)
horario_lectivo = (room_68.index.hour >= 8) & (room_68.index.hour <= 20) & (room_68.index.dayofweek < 5)

# Cruzamos las reglas: Sensor muerto + Invierno (Feb) + Clase = Inyectar Calor
falla_invierno = filtro_sensor_muerto & (room_68.index.month == 2) & horario_lectivo
# Sensor muerto + Verano (Jun) + Clase = Inyectar Frío
falla_verano = filtro_sensor_muerto & (room_68.index.month == 6) & horario_lectivo

num_muertos = filtro_sensor_muerto.sum()
print(f"⚠️ REPARACIÓN DE TELEMETRÍA (Pérdida de 'Heartbeat'):")
print(f"   Detectados {num_muertos} registros sin consumo base (dif_cons = 0.0).")
print(f"   -> Reparados {falla_invierno.sum()} huecos en horario lectivo de Invierno.")
print(f"   -> Reparados {falla_verano.sum()} huecos en horario lectivo de Verano.\n")

# Inyectamos las potencias calculadas en esos huecos
room_68.loc[falla_invierno, 'Q_hvac'] = potencia_media_calor
room_68.loc[falla_verano, 'Q_hvac'] = potencia_media_frio
# ==============================================================================

# 🚨 CAMBIO 2: MODIFICACIÓN DEL DROPNA 🚨
# Quitamos 'Q_hvac' de la lista del dropna. Si el sensor estaba muerto pero era de noche, 
# se queda el 0, lo cual es correcto. Si era de día, ya le hemos puesto el calor/frío.
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'radmed']).copy()


# Valores típicos de R y C
R_inicial = 0.02
C_inicial = 80000000
Asol_inicial = 2

col_T_int = 'V2'
col_T_ext = 'tmed'

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

# Calculo de error absoluto
datos_limpios['error_abs'] = (datos_limpios[col_T_int] - datos_limpios['T_simulada']).abs()

fin_simulacion = time.perf_counter()
tiempo_ejecucion = fin_simulacion - inicio_simulacion

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

plt.title(f'Modelo 1R1C: R={R_inicial}, C={C_inicial}')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()