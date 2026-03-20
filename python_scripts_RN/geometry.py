import numpy as np

def get_penrose_coords(t, rs, region, kp):
    """
    Mapea el tiempo 't' y la coordenada tortuga 'rs' a las coordenadas 
    compactificadas (R, T) del diagrama de Penrose según la región topológica.
    
    Parámetros:
    - t: array o float, coordenada temporal
    - rs: array o float, coordenada tortuga r*
    - region: string ('I', 'II', 'III', 'II_WH', 'III_WH')
    - kp: float, gravedad superficial del horizonte exterior
    """
    u = t - rs
    v = t + rs
    
    if region == 'I':
        # Universo Exterior (Agujero Negro, Diamante derecho)
        U = -np.exp(-np.clip(kp * u, -100, 100))
        V = np.exp(np.clip(kp * v, -100, 100))
        T = (np.arctan(V) + np.arctan(U)) / 2.0
        R = (np.arctan(V) - np.arctan(U)) / 2.0
        return R, T
        
    elif region == 'II':
        # Interior del Agujero Negro (Diamante superior central)
        U = np.exp(-np.clip(kp * u, -100, 100))
        V = np.exp(np.clip(kp * v, -100, 100))
        T = (np.arctan(V) + np.arctan(U)) / 2.0
        R = (np.arctan(V) - np.arctan(U)) / 2.0
        return R, T
        
    elif region == 'III':
        # Singularidad Futura (Triángulo superior izquierdo)
        V_tilde = np.arctan(np.exp(np.clip(kp * v, -100, 100)))
        U_tilde = np.pi - np.arctan(np.exp(np.clip(-kp * u, -100, 100)))
        T = (V_tilde + U_tilde) / 2.0
        R = (V_tilde - U_tilde) / 2.0
        return R, T
        
    elif region == 'II_WH':
        # Interior del Agujero Blanco (Diamante inferior central)
        # Reflexión puntual completa (-R, -T)
        U = np.exp(-np.clip(kp * u, -100, 100))
        V = np.exp(np.clip(kp * v, -100, 100))
        T = (np.arctan(V) + np.arctan(U)) / 2.0
        R = (np.arctan(V) - np.arctan(U)) / 2.0
        return -R, -T
        
    elif region == 'III_WH':
        # Singularidad Pasada (Triángulo inferior derecho)
        # Reflexión puntual completa (-R, -T)
        V_tilde = np.arctan(np.exp(np.clip(kp * v, -100, 100)))
        U_tilde = np.pi - np.arctan(np.exp(np.clip(-kp * u, -100, 100)))
        T = (V_tilde + U_tilde) / 2.0
        R = (V_tilde - U_tilde) / 2.0
        return -R, -T
        
    return 0, 0