import numpy
import pandas
from scipy.optimize import minimize
import time

# Carga y limpieza (Mantenemos tu lógica)
csv = pandas.read_csv('data-roomA-10T.csv', sep=';')
csv.columns = csv.columns.str.strip()
csv['Date'] = pandas.to_datetime(csv['Date'], utc=True)
csv.set_index('Date', inplace=True)

room_68 = csv[csv['room'] == 68].copy()
room_68.sort_index(inplace=True)

estado_HVAC = [room_68['V5_0'] == 1, room_68['V5_1'] == 1, room_68['V5_2'] == 1]
room_68['hvac_signo'] = numpy.select(estado_HVAC, [0, 1, -1])
bloqueA_rooms = csv['room'].nunique()
datos_limpios = room_68.dropna(subset=['V2', 'tmed', 'dif_cons', 'radmed']).copy()

def simulacion_1R1C(T_ext, dif_cons, hvac_signo, Rad_solar, T_int_inicial, R, C, A_sol, COP_fijo, dt_minutos=10):
    dt_segundos = dt_minutos * 60
    n_pasos = len(T_ext)
    T_sim = numpy.zeros(n_pasos)
    T_sim[0] = T_int_inicial
    
    T_ext_vals = T_ext.values
    dif_cons_vals = dif_cons.values
    hvac_vals = hvac_signo.values
    Rad_vals = Rad_solar.values
    
    for i in range(n_pasos - 1):
        flujo_paredes = (T_ext_vals[i] - T_sim[i]) / R
        Q_sol = Rad_vals[i] * A_sol
        
        # Potencia térmica con COP fijo
        Q_hvac_i = (dif_cons_vals[i] * 6000 / bloqueA_rooms) * COP_fijo * hvac_vals[i]
        
        dT = (flujo_paredes + Q_hvac_i + Q_sol) / C
        T_sim[i+1] = T_sim[i] + (dT * dt_segundos)
        
    return T_sim

def optimizar_RC_fijo(df, cop_valor=3.0, asol_valor=0.0):
    print(f"🚀 Optimizando R y C con COP fijo = {cop_valor} y A_sol = {asol_valor}")
    
    T_ext = df['tmed']
    dif_cons = df['dif_cons']
    hvac_signo = df['hvac_signo']
    Rad_solar = df['radmed']
    T_real = df['V2']
    T_int_inicial = T_real.iloc[0]

    def funcion_coste(params):
        R, C = params
        T_sim = simulacion_1R1C(
            T_ext, dif_cons, hvac_signo, Rad_solar, T_int_inicial, 
            R, C, asol_valor, cop_valor
        )
        return numpy.mean((T_real - T_sim)**2)

    # Semillas iniciales y límites realistas para evitar valores absurdos
    params_ini = [0.05, 8e7]
    limites = [
        (0.001, 0.1),  # R: De edificio poco aislado a muy aislado
        (1e7, 1.5e8)   # C: Rango lógico para masas de hormigón
    ]

    resultado = minimize(
        funcion_coste, 
        params_ini, 
        method='L-BFGS-B', 
        bounds=limites
    )
    
    R_opt, C_opt = resultado.x
    
    print("\n" + "="*40)
    print(f"R (Resistencia):  {R_opt:.6f} K/W")
    print(f"C (Capacitancia): {C_opt:,.0f} J/K")
    print(f"Tau (Inercia):    {R_opt * C_opt / 3600:.2f} horas")
    print("="*40)
    
    T_final = simulacion_1R1C(T_ext, dif_cons, hvac_signo, Rad_solar, T_int_inicial, R_opt, C_opt, asol_valor, cop_valor)
    return T_final, R_opt, C_opt

# Ejecución
datos_limpios['T_sim_opt'], r_opt, c_opt = optimizar_RC_fijo(datos_limpios, cop_valor=3.0)