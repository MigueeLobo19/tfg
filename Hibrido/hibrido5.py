import pandas 
import numpy
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time

# --- 1. CARGA Y PREPARACIÓN DE DATOS ---
print("Cargando y procesando datos...")
csv = pandas.read_csv('data-roomA-10T.csv', sep=';')

csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

# Filtramos la sala 68 y ordenamos
room_68 = csv[csv['room'] == 68].copy()
room_68.sort_index(inplace=True)

# Número de salas en el Bloque A para repartir el consumo
bloqueA_rooms = csv['room'].nunique()


# Lógica del HVAC (0 = Apagado, 1 = Calefacción, -1 = Frío)
estado_HVAC = [room_68['V5_0'] == 1, room_68['V5_1'] == 1, room_68['V5_2'] == 1]
multiplicadores = [0, 1, -1]
room_68['hvac'] = numpy.select(estado_HVAC, multiplicadores)

# Cálculo del Calor del HVAC (W)
COP_estimado = 4.5
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Dataset limpio sin nulos
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# --- 2. MODELO FÍSICO (LÍNEA BASE ESTÁTICA) ---
print("Ejecutando simulación física 1R1C...")
R_fija = 0.02
C_fija = 80000000
Asol_fijo = 2.0

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

# --- 3. IA DE RESIDUOS (SVR MEJORADO) ---
print("Entrenando IA para corregir los errores de la física...")
inicio_entrenamiento = time.perf_counter()

# Calculamos el error que la física no supo explicar
datos_limpios['residuo_fisico'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']

# Extraemos el tiempo (Feature Engineering)
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek
datos_limpios['mes'] = datos_limpios.index.month 

# Aprende en primavera-verano
print("\nFiltrando por estaciones...")
# Abril y mayo entrena
filtro_primavera = datos_limpios['mes'].isin([4, 5])
# Enero y febrero test
filtro_invierno = datos_limpios['mes'].isin([1, 2])  


# NUEVAS VARIABLES: La IA ahora ve el clima, el HVAC y lo que dijo la física
X = datos_limpios[['tmed', 'hvac', 'hora', 'dia_semana', 'radmed', 'T_simulada']]
y = datos_limpios['residuo_fisico']

# Escalado de datos (Crucial para el SVR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamiento del modelo SVR (C aumentada para mejor ajuste de curvas complejas)
modelo_corrector = SVR(kernel='rbf', C=100, epsilon=0.1)
modelo_corrector.fit(X_scaled, y)
fin_entrenamiento = time.perf_counter()

inicio_prediccion = time.perf_counter()
# La IA predice el error y se lo sumamos a la física para obtener la temperatura final
datos_limpios['correccion_IA'] = modelo_corrector.predict(X_scaled)
datos_limpios['T_hibrida'] = datos_limpios['T_simulada'] + datos_limpios['correccion_IA']
fin_prediccion = time.perf_counter()

# --- 4. MÉTRICAS Y GRÁFICA ---
T_real = datos_limpios[col_T_int]
T_hib = datos_limpios['T_hibrida']
T_fisica = datos_limpios['T_simulada']

# Cálculo del R2 Híbrido
r2_final = 1 - (numpy.sum((T_real - T_hib)**2) / numpy.sum((T_real - numpy.mean(T_real))**2))
# Cálculo del R2 de la Física pura (para comparar)
r2_fisica = 1 - (numpy.sum((T_real - T_fisica)**2) / numpy.sum((T_real - numpy.mean(T_real))**2))

mae_final = numpy.mean(numpy.abs(T_real - T_hib))

print("\n" + "=" * 50)
print("🎯 RESULTADOS DEL MODELO HÍBRIDO")
print("=" * 50)
print(f"R2 Física (Base): {r2_fisica:.4f}")
print(f"R2 Híbrido (Final): {r2_final:.4f}  <-- ¡Esta es la mejora de la IA!")
print(f"MAE Híbrido Final: {mae_final:.3f} °C")
print("-" * 50)
print("⚙️ COSTE COMPUTACIONAL (IA)")
print(f"Tiempo Entrenamiento : {fin_entrenamiento - inicio_entrenamiento:.4f} segundos")
print(f"Tiempo Predicción    : {(fin_prediccion - inicio_prediccion) * 1000:.2f} milisegundos")
print("=" * 50)

# Gráfica comparativa
plt.figure(figsize=(16, 7))
plt.plot(datos_limpios.index, T_real, label='Real (Sensor)', color='black', alpha=0.7, linewidth=1.5)
plt.plot(datos_limpios.index, datos_limpios['T_simulada'], label='Física Pura (1R1C)', color='orange', linestyle='--', alpha=0.6)
plt.plot(datos_limpios.index, T_hib, label='Híbrido (Física + IA)', color='red', linewidth=1.5)
plt.title('Gemelo Digital Híbrido: Sala 68 (Corrección de Residuo por SVR)', fontsize=14)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()