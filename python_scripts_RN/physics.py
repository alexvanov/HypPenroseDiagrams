import numpy as np
from scipy.optimize import fsolve

def get_bh_params(Mass, Q):
    """
    Calcula los parámetros geométricos del agujero negro de Reissner-Nordström.
    Devuelve los horizontes (rp, rm), la gravedad superficial (kp) y el offset de r*.
    """
    rp = Mass + np.sqrt(Mass**2 - Q**2)
    rm = Mass - np.sqrt(Mass**2 - Q**2)
    kp = (rp - rm) / (2 * rp**2)
    
    coef_p_offset = rp**2 / (rp - rm)
    coef_m_offset = rm**2 / (rp - rm)
    R_STAR_OFFSET = -(coef_p_offset * np.log(rp) - coef_m_offset * np.log(rm))
    
    return rp, rm, kp, R_STAR_OFFSET

def r_star_func(r, Mass, Q, rp, rm, R_STAR_OFFSET):
    """Calcula la coordenada tortuga r* para Reissner-Nordström."""
    coef_p = rp**2 / (rp - rm)
    coef_m = rm**2 / (rp - rm)
    rs = r + coef_p * np.log(np.abs(r - rp)) - coef_m * np.log(np.abs(r - rm))
    return rs + R_STAR_OFFSET

def dh_dr_CMC(h, r, M, Q, K, C):
    """Ecuación diferencial de la función de altura h(r) para foliaciones CMC."""
    f = 1.0 - (2.0 * M / r) + (Q**2 / r**2)
    term_K = (K * r / 3.0) + (C / r**2)
    disc = f + term_K**2
    if disc < 1e-12: 
        disc = 1e-12
    return -term_K / (f * np.sqrt(disc))

def get_trumpet_params(K_val, M_val, Q_val):
    """Encuentra numéricamente el valor C_CMC crítico y el radio de la trompeta R0."""
    def equations(vars):
        r, C = vars
        f = 1.0 - 2.0*M_val/r + (Q_val**2)/(r**2)
        df = 2.0*M_val/(r**2) - 2.0*(Q_val**2)/(r**3)
        term = (K_val * r / 3.0) + (C / (r**2))
        dterm = (K_val / 3.0) - (2.0 * C / (r**3))
        return [f + term**2, df + 2.0 * term * dterm]
    
    abs_K = abs(K_val)
    if abs_K < 0.2: initial_guess = [1.18, 0.1]
    elif abs_K < 0.6: initial_guess = [1.25, 0.5]
    else: initial_guess = [1.3, 1.0]

    sol = fsolve(equations, initial_guess)
    return float(sol[1]), float(sol[0])

def analizar_raices(C_val, K_val=-1.0, M_val=1.0, Q_val=0.9):
    """
    Encuentra las raíces reales positivas del discriminante del polinomio.
    Incluye la tolerancia para ruido numérico (evita el fallo con C_CMC = 1).
    """
    coeff = [K_val**2 / 9.0, 0.0, 1.0, (2.0 * K_val * C_val / 3.0) - 2.0 * M_val, Q_val**2, 0.0, C_val**2]
    todas_las_raices = np.roots(coeff)
    raices_reales = todas_las_raices[np.abs(todas_las_raices.imag) < 1e-10].real
    raices_positivas = sorted([r for r in raices_reales if r > 0])
    return raices_positivas