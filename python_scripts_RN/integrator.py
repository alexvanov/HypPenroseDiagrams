import numpy as np
from scipy.integrate import odeint

# Importamos las funciones matemáticas del módulo de physics
from physics import dh_dr_CMC, r_star_func, get_trumpet_params, analizar_raices

def integrar_foliacion(K_cmc, Mass, Q, condicion_C, rp, rm, R_STAR_OFFSET):
    """
    Controla el flujo lógico y resuelve las ecuaciones diferenciales de la foliación.
    Devuelve un diccionario estructurado con todos los datos necesarios para plotear.
    """
    EPS = 1e-8
    MARGIN_R0 = 1e-4

    data = {
        'regimen': None,
        'C_val': None,
        'raices': None,
        'tramos': {}
    }

    if condicion_C == 'critico':
        C_crit, R0_trumpet = get_trumpet_params(K_cmc, Mass, Q)
        print(f"-> Régimen CRÍTICO activado: C={C_crit:.5f}, R0={R0_trumpet:.5f}")
        
        data['regimen'] = 'critico'
        data['C_val'] = C_crit
        data['raices'] = R0_trumpet

        # --- A. Exterior (r > r+) ---
        r_phys_out = np.geomspace(rp + EPS, 300.0, 3000)
        h_vals_raw = odeint(dh_dr_CMC, 0.0, r_phys_out, args=(Mass, Q, K_cmc, C_crit)).flatten()
        offset = h_vals_raw[-1] - r_phys_out[-1]
        h_vals_out = h_vals_raw - offset
        rs_out = r_star_func(r_phys_out, Mass, Q, rp, rm, R_STAR_OFFSET)

        # --- B. Interior (R0 < r < r+) ---
        r_phys_in = np.geomspace(rp - EPS, R0_trumpet + MARGIN_R0, 2000)
        h_start_in = float(h_vals_out[ 0 ])
        h_vals_in = odeint(dh_dr_CMC, h_start_in, r_phys_in, args=(Mass, Q, K_cmc, C_crit)).flatten()
        rs_in = r_star_func(r_phys_in, Mass, Q, rp, rm, R_STAR_OFFSET)

        # --- C. Interior 2 (r- < r < R0) ---
        r_phys_in2 = np.linspace(R0_trumpet - MARGIN_R0, rm + EPS, 2000)
        h_vals_in2 = odeint(dh_dr_CMC, 0.0, r_phys_in2, args=(Mass, Q, K_cmc, C_crit)).flatten()
        rs_in2 = r_star_func(r_phys_in2, Mass, Q, rp, rm, R_STAR_OFFSET)

        # --- D. Interior 3 (0 < r < r-) ---
        delta_h_horizon = h_vals_in2[-1] + rs_in2[-1]
        r_phys_in3 = np.linspace(rm - EPS, EPS, 2000)
        rs_in3 = r_star_func(r_phys_in3, Mass, Q, rp, rm, R_STAR_OFFSET)
        h_start_in3 = delta_h_horizon - rs_in3[ 0 ]
        h_vals_in3 = odeint(dh_dr_CMC, h_start_in3, r_phys_in3, args=(Mass, Q, K_cmc, C_crit)).flatten()

        data['tramos'] = {
            'out': (r_phys_out, h_vals_out, rs_out),
            'in1': (r_phys_in, h_vals_in, rs_in),
            'in2': (r_phys_in2, h_vals_in2, rs_in2),
            'in3': (r_phys_in3, h_vals_in3, rs_in3)
        }

    else:
        C_val = float(condicion_C)
        data['C_val'] = C_val
        raices_positivas = analizar_raices(C_val, K_val=K_cmc, M_val=Mass, Q_val=Q)

        if len(raices_positivas) == 0:
            print(f"-> 0 raíces encontradas: RÉGIMEN SUPERCRÍTICO (C={C_val:.2f})")
            data['regimen'] = 'supercritico'

            # A. Exterior
            r_phys_out = np.geomspace(rp + EPS, 300.0, 2000)
            h_vals_raw = odeint(dh_dr_CMC, 0.0, r_phys_out, args=(Mass, Q, K_cmc, C_val)).flatten()
            offset = h_vals_raw[-1] - r_phys_out[-1]
            h_vals_out = h_vals_raw - offset
            rs_out = r_star_func(r_phys_out, Mass, Q, rp, rm, R_STAR_OFFSET)

            # B. Interior 1
            r_phys_in = np.linspace(rp - EPS, rm + EPS, 2000)
            rs_in = r_star_func(r_phys_in, Mass, Q, rp, rm, R_STAR_OFFSET)
            h_start_in = float(h_vals_out[ 0 ]) + float(rs_out[ 0 ]) - float(rs_in[ 0 ]) 
            h_vals_in = odeint(dh_dr_CMC, h_start_in, r_phys_in, args=(Mass, Q, K_cmc, C_val)).flatten()

            # C. Interior 2
            r_phys_in2 = np.linspace(rm - EPS, EPS, 2000)
            rs_in2 = r_star_func(r_phys_in2, Mass, Q, rp, rm, R_STAR_OFFSET)
            h_start_in2 = float(h_vals_in[-1]) + float(rs_in[-1]) - float(rs_in2[ 0 ]) 
            h_vals_in2 = odeint(dh_dr_CMC, h_start_in2, r_phys_in2, args=(Mass, Q, K_cmc, C_val)).flatten()

            data['tramos'] = {
                'out': (r_phys_out, h_vals_out, rs_out),
                'in1': (r_phys_in, h_vals_in, rs_in),
                'in2': (r_phys_in2, h_vals_in2, rs_in2)
            }

        elif len(raices_positivas) >= 2:
           
            R1 = max(raices_positivas[ 0 ], rm + 1e-5) 
            R2 = min(raices_positivas[ 1 ], rp - 1e-5) 
            print(f"-> 2 raíces encontradas: RÉGIMEN SUBCRÍTICO (R1={R1:.3f}, R2={R2:.3f})")
            
            data['regimen'] = 'subcritico'
            data['raices'] = (R1, R2)

            # A. Exterior
            r_phys_out = np.geomspace(rp + EPS, 300.0, 2000)
            h_vals_raw = odeint(dh_dr_CMC, 0.0, r_phys_out, args=(Mass, Q, K_cmc, C_val)).flatten()
            offset = h_vals_raw[-1] - r_phys_out[-1]
            h_vals_out = h_vals_raw - offset
            rs_out = r_star_func(r_phys_out, Mass, Q, rp, rm, R_STAR_OFFSET)

            # B. Tramo diamante superior
            r_phys_in_R2 = np.linspace(rp - EPS, R2 + EPS, 2000)
            rs_in_R2 = r_star_func(r_phys_in_R2, Mass, Q, rp, rm, R_STAR_OFFSET)
            h_start_in_R2 = float(h_vals_out[ 0 ]) + float(rs_out[ 0 ]) - float(rs_in_R2[ 0 ]) 
            h_vals_in_R2 = odeint(dh_dr_CMC, h_start_in_R2, r_phys_in_R2, args=(Mass, Q, K_cmc, C_val), mxstep=50000).flatten()

            # C. Tramo diamante inferior
            r_phys_in_R1 = np.linspace(R1 - EPS, rm + EPS, 2000)
            rs_in_R1 = r_star_func(r_phys_in_R1, Mass, Q, rp, rm, R_STAR_OFFSET)
            h_start_in_R1 = 0.0 
            h_vals_in_R1 = odeint(dh_dr_CMC, h_start_in_R1, r_phys_in_R1, args=(Mass, Q, K_cmc, C_val), mxstep=50000).flatten()

            # D. Tramo singularidad
            r_phys_in3 = np.linspace(rm - EPS, EPS, 2000)
            rs_in3 = r_star_func(r_phys_in3, Mass, Q, rp, rm, R_STAR_OFFSET)
            h_start_in3 = float(h_vals_in_R1[-1]) + float(rs_in_R1[-1]) - float(rs_in3[ 0 ])
            h_vals_in3 = odeint(dh_dr_CMC, h_start_in3, r_phys_in3, args=(Mass, Q, K_cmc, C_val), mxstep=50000).flatten()

            data['tramos'] = {
                'out': (r_phys_out, h_vals_out, rs_out),
                'in_R2': (r_phys_in_R2, h_vals_in_R2, rs_in_R2),
                'in_R1': (r_phys_in_R1, h_vals_in_R1, rs_in_R1),
                'in3': (r_phys_in3, h_vals_in3, rs_in3)
            }

    return data