import pandas 
import numpy
import matplotlib.pyplot as plt 
import time

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
#room_68['dif_cons_limpio'] = room_68['dif_cons'].clip(upper=25.0)
#room_68['dif_cons_suavizado'] = room_68['dif_cons_limpio'].rolling(window=3, min_periods=1).mean()
room_68['P_electrica_W'] = (room_68['dif_cons'] * 6 * 1000) / bloqueA_rooms
room_68['Q_hvac'] = room_68['P_electrica_W'] * COP_estimado * room_68['hvac']
room_68['Q_hvac'] = room_68['Q_hvac'].clip(lower=-3500.0, upper=3500.0)

datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'Q_hvac']).copy()

datos_limpios['hora'] = datos_limpios.index.hour
datos_limpios['dia_semana'] = datos_limpios.index.dayofweek 
datos_limpios['mes'] = datos_limpios.index.month 

columnas_X = ['tmed', 'radmed', 'hvac', 'hora', 'dia_semana']
X = datos_limpios[columnas_X]
y = datos_limpios['V2'] 


# Aprende en primavera-verano
print("\nFiltrando por estaciones...")
# Abril y mayo entrena
filtro_primavera = datos_limpios['mes'].isin([4, 5])
# Enero y febrero test
filtro_invierno = datos_limpios['mes'].isin([1, 2])  

X_train = X[filtro_primavera]
y_train = y[filtro_primavera]

X_test = X[filtro_invierno]
y_test = y[filtro_invierno]

print(f"Datos de entrenamiento: {len(X_train)} horas")
print(f"Datos de prueba: {len(X_test)} horas")

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Error: No hay suficientes datos en los meses seleccionados para hacer la prueba.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.1) 

print("\nEntrenando en primavera-verano")
inicio_entrenamiento = time.perf_counter()
modelo_svr.fit(X_train_scaled, y_train)
fin_entrenamiento = time.perf_counter()

# Predicción en invierno
print("Predicción en invierno")
inicio_prediccion = time.perf_counter()
y_pred_test = modelo_svr.predict(X_test_scaled)
fin_prediccion = time.perf_counter()

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = numpy.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(" RESULTADOS DE TEST EN INVIERNO")
print(f"MAE: {mae:.3f} °C")
print(f"RMSE: {rmse:.3f} °C")
print(f"R²: {r2:.4f}\n")

print(f"Tiempo Entrenamiento : {fin_entrenamiento - inicio_entrenamiento:.4f} seg")



plt.figure(figsize=(15, 7))


plt.plot(y_test.index, y_test, label='T. Interior REAL (Invierno)', color='black', linewidth=1.5)
plt.plot(y_test.index, y_pred_test, label='Predicción SVR (Cree que es Primavera)', color='red', linestyle='--')
plt.plot(y_test.index, X_test['tmed'], label='T. Exterior (Frío)', color='blue', alpha=0.3)

plt.title('Resultados entrenamiento en distintas épocas del año')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()