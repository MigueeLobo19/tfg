# ==============================================================================
# EXPERIMENTO TFG: PRUEBA DE EXTRAPOLACIÓN ESTACIONAL EN MODELO GREY-BOX
# ==============================================================================
import pandas 
import numpy
import matplotlib.pyplot as plt 
import time
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print(" PRUEBA DE ESTRÉS: GEMELO DIGITAL (PRIMAVERA -> INVIERNO)")
print("=" * 55)

# --- 1. LECTURA Y LIMPIEZA DE DATOS ---
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
room_68['dif_cons_limpio'] = room_68['dif_cons'].clip(upper=25.0)
room_68['dif_cons_suavizado'] = room_68['dif_cons_limpio'].rolling(window=3, min_periods=1).mean()
room_68['P_electrica_W'] = (room_68['dif_cons_suavizado'] * 6 * 1000) / bloqueA_rooms
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']
room_68['Q_hvac'] = room_68['Q_hvac'].clip(lower=-3500.0, upper=3500.0)

datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()
datos_limpios['mes'] = datos_limpios.index.month 

# --- 2. DIVISIÓN ESTACIONAL (Train y Test) ---
filtro_primavera = datos_limpios['mes'].isin([4, 5]) # Abril y Mayo (Calor)
filtro_invierno = datos_limpios['mes'].isin([1, 2])  # Enero y Febrero (Frío)

# Partimos el dataframe en dos conjuntos separados
df_train = datos_limpios[filtro_primavera].copy()
df_test = datos_limpios[filtro_invierno].copy()

print(f"Datos para Optimizar (Primavera) : {len(df_train)} registros")
print(f"Datos para Simular   (Invierno)  : {len(df_test)} registros")

# --- 3. FUNCIONES DEL MODELO FÍSICO HÍBRIDO ---
def simulacion_1R1C_sol(T_ext, Q_hvac, Rad, T_int_inicial, R, C, A_sol, dt_minutos=10):
    dt_segundos = dt_minutos * 60
    n_pasos = len(T_ext)
    T_sim = numpy.zeros(n_pasos)
    T_sim[0] = T_int_inicial
    
    T_ext_vals = T_ext.values
    Q_hvac_vals = Q_hvac.values
    Rad_vals = Rad.values
    
    for i in range(n_pasos - 1):
        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        flujo_solar = Rad_vals[i] * A_sol
        dT = (flujo_paredes + Q_hvac_vals[i] + flujo_solar) / C
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
    return T_sim

def error_optimizacion(params, T_ext, Q_hvac, Rad, T_real, T_int_inicial):
    R, C, A_sol = params
    T_sim = simulacion_1R1C_sol(T_ext, Q_hvac, Rad, T_int_inicial, R, C, A_sol)
    return numpy.mean((T_real - T_sim)**2) # MSE como función de coste

# --- 4. FASE DE ENTRENAMIENTO (OPTIMIZACIÓN EN PRIMAVERA) ---
print("\nEjecutando Optimizador (L-BFGS-B) solo en Primavera...")
parametros_iniciales = [0.005, 60000000, 2.0]
limites = [(0.0001, 0.1), (1000000, 100000000), (0.0, 50.0)]

inicio_opt = time.perf_counter()
resultado_opt = minimize(
    error_optimizacion,
    parametros_iniciales,
    args=(df_train['tmed'], df_train['Q_hvac'], df_train['radmed'], df_train['V2'], df_train['V2'].iloc[0]),
    method='L-BFGS-B',
    bounds=limites
)
fin_opt = time.perf_counter()

R_opt, C_opt, Asol_opt = resultado_opt.x

print(f"Parámetros descubiertos en Primavera:")
print(f" - R (Resistencia) : {R_opt:.5f}")
print(f" - C (Inercia)     : {C_opt:.0f}")
print(f" - A_sol (Ventana) : {Asol_opt:.2f} m²")

# --- 5. FASE DE PRUEBA (SIMULACIÓN CIEGA EN INVIERNO) ---
print("\nSimulando el Invierno con la R y C descubiertas...")
inicio_sim = time.perf_counter()

# Usamos los datos de Invierno, pero los parámetros (R, C, Asol) del verano!
T_sim_invierno = simulacion_1R1C_sol(
    T_ext = df_test['tmed'],
    Q_hvac = df_test['Q_hvac'],
    Rad = df_test['radmed'],
    T_int_inicial = df_test['V2'].iloc[0],
    R = R_opt,
    C = C_opt,
    A_sol = Asol_opt
)
fin_sim = time.perf_counter()

df_test['T_simulada'] = T_sim_invierno

# --- 6. MÉTRICAS DE RENDIMIENTO (SOBRE EL INVIERNO) ---
T_real_test = df_test['V2']

mae = mean_absolute_error(T_real_test, T_sim_invierno)
mse = mean_squared_error(T_real_test, T_sim_invierno)
rmse = numpy.sqrt(mse)
r2 = r2_score(T_real_test, T_sim_invierno)

print("\n" + "=" * 55)
print(" RESULTADOS GEMELO DIGITAL (PREDICCIÓN EN INVIERNO)")
print("=" * 55)
print(f"MAE  : {mae:.3f} °C")
print(f"RMSE : {rmse:.3f} °C")
print(f"R²   : {r2:.4f}")
print("-" * 55)
print(f"Tiempo Optimización (Primavera) : {fin_opt - inicio_opt:.4f} seg")
print(f"Tiempo Simulación (Invierno)    : {(fin_sim - inicio_sim) * 1000:.2f} ms")
print("=" * 55)

# --- 7. GRÁFICA COMPARATIVA ---
plt.figure(figsize=(15, 7))

plt.plot(df_test.index, df_test['V2'], label='T. Interior REAL (Invierno)', color='black', linewidth=1.5)
plt.plot(df_test.index, df_test['T_simulada'], label='T. Simulada (Grey-Box entrenado en Primavera)', color='green', linestyle='--')
plt.plot(df_test.index, df_test['tmed'], label='T. Exterior (Frío)', color='blue', alpha=0.3)

plt.title('Prueba de Robustez Estacional: Modelo Grey-Box (1R1C Optimizado)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()