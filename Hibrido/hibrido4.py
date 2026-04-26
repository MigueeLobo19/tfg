import pandas 
import numpy
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para calcular el 1R1C
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Valores típicos de R y C
R_inicial = 0.005
C_inicial = 60000000
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
    A_sol = 2.5  # El valor numérico fijo (en este ejemplo, 2.5 metros cuadrados)
)


# Guardamos el resultado
datos_limpios['T_simulada'] = T_simulada
datos_limpios['residuo_fisico'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek

# --- 3. DIVISIÓN ESTACIONAL (El truco está aquí) ---
# Definimos los meses de cada estación
meses_primavera = [3, 4, 5]   # Marzo, Abril, Mayo (Entrenamiento)
meses_invierno = [12, 1, 2]   # Diciembre, Enero, Febrero (Examen/Test)

datos_train = datos_limpios[datos_limpios.index.month.isin(meses_primavera)].copy()
datos_test = datos_limpios[datos_limpios.index.month.isin(meses_invierno)].copy()

# Aviso de seguridad por si el CSV no tiene esos meses
if len(datos_train) == 0 or len(datos_test) == 0:
    print("¡ADVERTENCIA! El dataset no contiene datos de los meses especificados.")
    print("Por favor, ajusta las listas 'meses_primavera' y 'meses_invierno' a los meses que sí existan en tu CSV.")

print(f"Entrenando IA con datos de PRIMAVERA ({len(datos_train)} registros)...")
X_train = datos_train[['hora', 'dia_semana', 'tmed']]
y_train = datos_train['residuo_fisico']

X_test = datos_test[['hora', 'dia_semana', 'tmed']]
y_test_real = datos_test['residuo_fisico'] # El error que la IA DEBERÍA predecir

# --- 4. ENTRENAMIENTO (Solo ve primavera) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

modelo_corrector = SVR(kernel='rbf', C=10, epsilon=0.01)
modelo_corrector.fit(X_train_scaled, y_train)

# --- 5. PREDICCIÓN (Se examina en invierno) ---
print(f"Evaluando IA con datos de INVIERNO ({len(datos_test)} registros)...")
# OJO: Usamos scaler.transform, NO fit_transform, para no cometer Data Leakage
X_test_scaled = scaler.transform(X_test)

datos_test['correccion_IA'] = modelo_corrector.predict(X_test_scaled)
datos_test['T_hibrida'] = datos_test['T_simulada'] + datos_test['correccion_IA']

# --- 6. MÉTRICAS Y GRÁFICA (Solo del periodo de Invierno) ---
T_real = datos_test[col_T_int]
T_hib = datos_test['T_hibrida']
T_fisica_pura = datos_test['T_simulada']

r2_final = 1 - (numpy.sum((T_real - T_hib)**2) / numpy.sum((T_real - numpy.mean(T_real))**2))
mae_final = numpy.mean(numpy.abs(T_real - T_hib))

r2_fisica = 1 - (numpy.sum((T_real - T_fisica_pura)**2) / numpy.sum((T_real - numpy.mean(T_real))**2))

print("\n" + "=" * 50)
print(" RESULTADOS: PRUEBA DE EXTRAPOLACIÓN INVERNAL")
print("=" * 50)
print(f"R2 (Solo Física)  : {r2_fisica:.4f}  <-- Lo que logra la ecuación diferencial sola")
print(f"R2 (Híbrido)      : {r2_final:.4f}  <-- Lo que logra tras la 'corrección' de la IA")
print(f"MAE Híbrido       : {mae_final:.3f} °C")
print("=" * 50)

# Gráfica para ver el desastre/éxito en Invierno
plt.figure(figsize=(15, 7))
plt.plot(datos_test.index, T_real, label='Real (Sensor en Invierno)', color='black', alpha=0.8)
plt.plot(datos_test.index, T_fisica_pura, label='Solo Física (1R1C)', color='blue', linestyle=':', alpha=0.5)
plt.plot(datos_test.index, T_hib, label='Híbrido (SVR entrenado en Primavera)', color='red', linewidth=1.5)
plt.title('Prueba de Estrés: IA entrenada en Primavera testeada en Invierno')
plt.ylabel('Temperatura Interior (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()