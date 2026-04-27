import pandas 
import numpy
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

print("Comienza la simulación del modelo híbrido y la evaluación de la generalización...\n")

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
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms

# Calculamos el calor térmico (Q)
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']

# Usamos los datos del dataset que usaremos para predecir con svr
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac', 'radmed']).copy()

# Comienza el modelo físico 
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

print("Entrenando modelo para corregir los errores del modelo 1R1C...")
inicio_entrenamiento = time.perf_counter()

# Calculamos el error
datos_limpios['residuo_fisico'] = datos_limpios[col_T_int] - datos_limpios['T_simulada']

# Hacemos que el modelo conozca el comportamiento, es decir, aprenda en función de la hora y dia de la semana
datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek
datos_limpios['mes'] = datos_limpios.index.month 

# Entradas y salidas del modelo
X = datos_limpios[['tmed', 'hvac', 'hora', 'dia_semana', 'radmed', 'T_simulada']]
y = datos_limpios['residuo_fisico']

print("\nFiltrando por estaciones...")
# Entrenamiento: Primavera (Abril y Mayo)
filtro_primavera = datos_limpios['mes'].isin([1, 2])
X_train = X[filtro_primavera]
y_train = y[filtro_primavera]

# Test: Invierno (Enero y Febrero)
filtro_invierno = datos_limpios['mes'].isin([8, 9])  
X_test = X[filtro_invierno]
y_test = y[filtro_invierno]

# Escalado para que las magnitudes se estandaricen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SVR
modelo_svr = SVR(kernel='rbf', C=10, epsilon=0.1)

print("\nEntrenando modelo SVR")
inicio_entrenamiento = time.perf_counter()
modelo_svr.fit(X_train_scaled, y_train)
fin_entrenamiento = time.perf_counter()
tiempo_train = fin_entrenamiento - inicio_entrenamiento

# Predicción sobre datos test (Invierno)
inicio_prediccion = time.perf_counter()
y_pred_test = modelo_svr.predict(X_test_scaled)
fin_prediccion = time.perf_counter()
tiempo_pred = fin_prediccion - inicio_prediccion

# Temperatura del modelo híbrido
T_real_test = datos_limpios.loc[y_test.index, col_T_int] 
T_fisica_test = X_test['T_simulada']                    
T_hib_test = T_fisica_test + y_pred_test               

# Cálculo métricas
mae = mean_absolute_error(T_real_test, T_hib_test)
mse = mean_squared_error(T_real_test, T_hib_test)
rmse = numpy.sqrt(mse)
r2 = r2_score(T_real_test, T_hib_test)

print("RESULTADOS DEL MODELO HÍBRIDO (Test en Invierno)")
print(f"MAE  (Error Medio Absoluto) : {mae:.3f} °C")
print(f"MSE  (Error Cuadrático Medio): {mse:.3f}")
print(f"RMSE (Raíz del MSE)         : {rmse:.3f} °C")
print(f"R2   (Coef. de Determinación): {r2:.4f}\n")
      
print("COSTE COMPUTACIONAL")
print(f"Tiempo Entrenamiento : {tiempo_train:.4f} segundos")
print(f"Tiempo Predicción    : {tiempo_pred:.2f} segundos")

# Gráfica
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, T_real_test, label='Temperatura interior real', color='black', linewidth=1.5)
plt.plot(y_test.index, T_fisica_test, label='Temperatura modelo 1R1C', color='green', linestyle='-.', alpha=0.6)
plt.plot(y_test.index, T_hib_test, label='Temperatura modelo híbrido', color='orange', linestyle='--')
plt.plot(y_test.index, X_test['tmed'], label='Temperatura exterior', color='blue', alpha=0.3)
plt.plot(y_test.index, y_pred_test, label='Predicción SVR', color='red', alpha=0.7, linestyle=':')

plt.title('Comparación: Modelo Físico vs Modelo Híbrido (Datos de Invierno)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()