import pandas 
import numpy
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time

# --- 1. PREPARACIÓN DE DATOS ---
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


# Salas en bloque A
bloqueA_rooms = csv['room'].nunique()

# Estado del HVAC
estado_HVAC = [room_68['V5_0'] == 1, room_68['V5_1'] == 1, room_68['V5_2'] == 1]

# En función de si el aporte calorífico es positivo (calefacción) o negativo (aire acondicionado)
multiplicadores = [0, 1, -1]

# Aplicamos los multiplicadores anteriores
room_68['hvac'] = numpy.select(estado_HVAC, multiplicadores)

# Definimos el rendimiento estimado del equipo 
COP_estimado = 4.5

# Calculamos la potencia electrica entre mediciones
# se divide entre el número de salas ya que el consumo se mide por bloque
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para calcular el 1R1C
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# --- 2. MODELO FÍSICO 1R1C ---
# Valores típicos de R y C
R_inicial = 0.02
C_inicial = 80000000
Asol_inicial = 2.0

col_T_int = 'V2'
col_T_ext = 'tmed'

inicio_simulacion = time.perf_counter()

# AÑADIMOS 'Rad_solar' A LOS PARÁMETROS DE ENTRADA
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
        # Flujo de calor por conducción (muros)
        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        
        # Flujo de calor por radiación (W/m2 * m2 = W)
        Q_sol = Rad_vals[i] * A_sol # <-- Multiplicamos el dato de ese instante por el tamaño de la ventana
        
        # Variación total de temperatura (la suma de todos los calores)
        dT = (flujo_paredes + Q_hvac_vals[i] + Q_sol) / C
        
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
    A_sol = Asol_inicial  # El valor numérico fijo
)

# Guardamos el resultado
datos_limpios['T_simulada'] = T_simulada

# --- 3. IA DE RESIDUOS ---
print("Entrenando IA para corregir los errores de la física estática...")
inicio_entrenamiento = time.perf_counter()

datos_limpios['residuo_fisico'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek

# Vector de características original (Culpable del R2 negativo)
X = datos_limpios[['hora', 'dia_semana', 'tmed']]
y = datos_limpios['residuo_fisico']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
modelo_corrector = SVR(kernel='rbf', C=10, epsilon=0.01)
modelo_corrector.fit(X_scaled, y)
fin_entrenamiento = time.perf_counter()

inicio_prediccion = time.perf_counter()
datos_limpios['correccion_IA'] = modelo_corrector.predict(X_scaled)
datos_limpios['T_hibrida'] = datos_limpios['T_simulada'] + datos_limpios['correccion_IA']
fin_prediccion = time.perf_counter()

# --- 4. MÉTRICAS Y GRÁFICA ---
T_real = datos_limpios[col_T_int]
T_hib = datos_limpios['T_hibrida']

r2_final = 1 - (numpy.sum((T_real - T_hib)**2) / numpy.sum((T_real - numpy.mean(T_real))**2))
mae_final = numpy.mean(numpy.abs(T_real - T_hib))

print("-" * 40)
print(f"R2 Híbrido Final: {r2_final:.4f}")
print(f"MAE Híbrido Final: {mae_final:.3f} °C")
print("-" * 40)
print("COSTE COMPUTACIONAL (IA)")
print(f"Tiempo Entrenamiento : {fin_entrenamiento - inicio_entrenamiento:.4f} segundos")
print(f"Tiempo Predicción    : {(fin_prediccion - inicio_prediccion) * 1000:.2f} milisegundos")

plt.figure(figsize=(15, 7))
plt.plot(datos_limpios.index, T_real, label='Real (Sensor)', color='black', alpha=0.6)
plt.plot(datos_limpios.index, T_hib, label='Híbrido (Física Estática + IA)', color='red', linewidth=2)
plt.title('Gemelo Digital: Física Estática compensada por IA')
plt.legend()
plt.show()