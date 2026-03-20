import matplotlib.pyplot as plt
import numpy as np
from geometry import get_penrose_coords
from physics import r_star_func

def configurar_estilo():
    """Configura el estilo de LaTeX para los gráficos."""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{amsmath}
    \usepackage{mathrsfs}
    \DeclareSymbolFontAlphabet{\mathrsfs}{rsfs}
    \newcommand{\scri}{\mathrsfs{I}}
    '''

def graficar_diagrama(data, Mass, Q, rp, rm, kp, R_STAR_OFFSET, extendido=False):
    """
    Toma los datos integrados y genera el diagrama de Penrose correspondiente
    dependiendo de si el régimen es crítico, subcrítico o supercrítico.
    """
    configurar_estilo()
    
    plt.figure(figsize=(10, 13), dpi=120)
    p4, p2 = np.pi/4, np.pi/2
    regimen = data['regimen']
    tramos = data['tramos']
    

    # 1. ESQUELETO Y LÍNEAS DE FONDO
    # Diamante Derecho, Superior y Triángulo Superior
    plt.plot([0, p4], [0, p4], 'k-', lw=1.5)       
    plt.plot([0, p4], [0, -p4], 'k-', lw=1.0)      
    plt.plot([p4, p2], [p4, 0], 'k-', lw=1.5)      
    plt.plot([p4, p2], [-p4, 0], 'k-', lw=1.5)     
    plt.plot([-p4, 0], [p4, p2], 'k-', lw=1.5)    
    plt.plot([0, -p4], [0, p4], 'k-', lw=1.0) 
    plt.plot([0, p4], [p2, p4], 'k-', lw=1.5)      
    plt.plot([0, -p4], [p2, 3*p4], 'k-', lw=1.0)   
    plt.plot([-p4, -p4], [p4, 3*p4], 'k-', lw=3.0) 

    if extendido == True:
        # Añadimos Diamante Izquierdo, Inferior y Triángulo Inferior 
        plt.plot([0, -p4], [0, p4], 'k-', lw=1.0)      
        plt.plot([0, -p4], [0, -p4], 'k-', lw=1.0)     
        plt.plot([-p4, -p2], [p4, 0], 'k-', lw=1.5)  
        plt.plot([0, p4], [0, -p4], 'k-', lw=1.0)
        plt.plot([0, p4], [0, p4], 'k-', lw=1.0)  
        plt.plot([-p4, -p2], [-p4, 0], 'k-', lw=1.5)   
        plt.plot([-p4, 0], [-p4, -p2], 'k-', lw=1.5)   
        plt.plot([0, p4], [-p2, -p4], 'k-', lw=1.5)    
        plt.plot([0, p4], [-p2, -3*p4], 'k-', lw=1.0) 
        plt.plot([p4, p4], [-p4, -3*p4], 'k-', lw=3.0)

    # Fondo r=constante
    t_bg = np.linspace(-300, 300, 2000)
    for r_val in [-10, 0, 1.8, 2.7, 4.5]:
        R_bg, T_bg = get_penrose_coords(t_bg, r_star_func(r_val, Mass, Q, rp, rm, R_STAR_OFFSET), 'I', kp)
        plt.plot(R_bg, T_bg, 'k--', lw=0.6, alpha=0.5)
        if regimen == 'subcritico' and extendido == True: plt.plot(-R_bg, T_bg, 'k--', lw=0.6, alpha=0.5) 
        
    for r_val in [-12, -8, 0, 1.8, 2.7, 4.5]:
        R_bg, T_bg = get_penrose_coords(t_bg, r_star_func(r_val, Mass, Q, rp, rm, R_STAR_OFFSET), 'II', kp)
        plt.plot(R_bg, T_bg, 'k--', lw=0.6, alpha=0.5)
        if regimen == 'subcritico' and extendido == True:
            R_bg_WH, T_bg_WH = get_penrose_coords(t_bg, r_star_func(r_val, Mass, Q, rp, rm, R_STAR_OFFSET), 'II_WH', kp)
            plt.plot(R_bg_WH, T_bg_WH, 'k--', lw=0.6, alpha=0.5)
            
    for r_val in [0.563, 3.9, 7]:
        R_bg, T_bg = get_penrose_coords(t_bg, r_star_func(r_val, Mass, Q, rp, rm, R_STAR_OFFSET), 'III', kp)
        plt.plot(R_bg, T_bg, 'k--', lw=0.6, alpha=0.5)
        if regimen == 'subcritico' and extendido == True:
            R_bg_WH, T_bg_WH = get_penrose_coords(t_bg, r_star_func(r_val, Mass, Q, rp, rm, R_STAR_OFFSET), 'III_WH', kp)
            plt.plot(R_bg_WH, T_bg_WH, 'k--', lw=0.6, alpha=0.5)

    
    # 2. SLICES PLOTTING
    t_escala = 1.0 / kp
    
    list_tau_out = np.linspace(-5 * t_escala, 8 * t_escala, 12)
    
    h_horizonte_ext = data['tramos']['out'][1][0]
    list_tau_in_critico = np.linspace(h_horizonte_ext - 2*t_escala, h_horizonte_ext + 10*t_escala, 10)
    
    list_tau_in_sub = np.linspace(-6 * t_escala, 6 * t_escala, 10)

    if regimen == 'subcritico':
        R1, R2 = data['raices']
        r_out, h_out, rs_out = tramos['out']
        r_R2, h_R2, rs_R2 = tramos['in_R2']
        r_R1, h_R1, rs_R1 = tramos['in_R1']
        r_in3, h_in3, rs_in3 = tramos['in3']

        # Slices Azules
        for tau in list_tau_out:
            R_I, T_I = get_penrose_coords(tau + h_out, rs_out, 'I', kp)
            R_II_R2, T_II_R2 = get_penrose_coords(tau + h_R2, rs_R2, 'II', kp)
            R_full_out = np.concatenate([R_II_R2[::-1], R_I])
            T_full_out = np.concatenate([T_II_R2[::-1], T_I])
            plt.plot(R_full_out, T_full_out, 'tab:blue', lw=1.2, alpha=0.9)
            if extendido == True:
                plt.plot(-R_full_out, -T_full_out, 'tab:blue', lw=1.2, alpha=0.9) 

        # Slices Naranjas
        for tau in list_tau_in_sub:
            R_II_R1, T_II_R1 = get_penrose_coords(tau + h_R1, rs_R1, 'II', kp)
            R_III, T_III = get_penrose_coords(tau + h_in3, rs_in3, 'III', kp)
            R_full_in = np.concatenate([R_III[::-1], R_II_R1[::-1]])
            T_full_in = np.concatenate([T_III[::-1], T_II_R1[::-1]])
            plt.plot(R_full_in, T_full_in, 'tab:orange', lw=1.2, alpha=0.9)
            if extendido == True:
                plt.plot(-R_full_in, -T_full_in, 'tab:orange', lw=1.2, alpha=0.9) 

        # 4 Muros Negros
        t_wall = np.linspace(-200, 200, 2000)
        rs_w2 = r_star_func(R2 * np.ones_like(t_wall), Mass, Q, rp, rm, R_STAR_OFFSET)
        R_w2, T_w2 = get_penrose_coords(t_wall, rs_w2, 'II', kp)
        plt.plot(R_w2, T_w2, 'k-', lw=2.5) 
        if extendido == True:
            plt.plot(-R_w2, -T_w2, 'k-', lw=2.5) 

        rs_w1 = r_star_func(R1 * np.ones_like(t_wall), Mass, Q, rp, rm, R_STAR_OFFSET)
        R_w1, T_w1 = get_penrose_coords(t_wall, rs_w1, 'II', kp)
        plt.plot(R_w1, T_w1, 'k-', lw=2.5) 
        if extendido == True:
            plt.plot(-R_w1, -T_w1, 'k-', lw=2.5) 

    else:
        # Lógica general para Crítico y Supercrítico
        r_out, h_out, rs_out = tramos['out']
        r_in1, h_in1, rs_in1 = tramos['in1']
        r_in2, h_in2, rs_in2 = tramos['in2']
        
        if regimen == 'critico':
            r_in3, h_in3, rs_in3 = tramos['in3']
            R0 = data['raices']
            for tau in list_tau_out:
                R_I, T_I = get_penrose_coords(tau + h_out, rs_out, 'I', kp)
                R_II_out, T_II_out = get_penrose_coords(tau + h_in1, rs_in1, 'II', kp)
                plt.plot(np.concatenate([R_II_out[::-1], R_I]), np.concatenate([T_II_out[::-1], T_I]), 'tab:blue', lw=1.2, alpha=0.9)
            list_tau_in = [26,30, 33,36,42] 
            for tau in list_tau_in_critico: 
                R_II_in, T_II_in = get_penrose_coords(tau + h_in2, rs_in2, 'II', kp)
                R_III, T_III = get_penrose_coords(tau + h_in3, rs_in3, 'III', kp)
                plt.plot(np.concatenate([R_III[::-1], R_II_in[::-1]]), np.concatenate([T_III[::-1], T_II_in[::-1]]), 'tab:orange', lw=1.2, alpha=0.9)

            # Pared Trompeta
            t_wall = np.linspace(-100, 200, 1000)
            rs_wall = r_star_func(R0 * np.ones_like(t_wall), Mass, Q, rp, rm, R_STAR_OFFSET)
            R_w, T_w = get_penrose_coords(t_wall, rs_wall, 'II', kp)
            plt.plot(R_w, T_w, 'k-', lw=2.5)

        elif regimen == 'supercritico':
            for tau in list_tau_out:
                R_I, T_I = get_penrose_coords(tau + h_out, rs_out, 'I', kp)
                R_II, T_II = get_penrose_coords(tau + h_in1, rs_in1, 'II', kp)
                R_III, T_III = get_penrose_coords(tau + h_in2, rs_in2, 'III', kp)
                R_full = np.concatenate([R_III[::-1], R_II[::-1], R_I])
                T_full = np.concatenate([T_III[::-1], T_II[::-1], T_I])
                plt.plot(R_full, T_full, 'tab:blue', lw=1.2, alpha=0.9)

    # ETIQUETAS
    if extendido == True and regimen == 'subcritico':
        plt.xlim(-p2 - 0.4, p2 + 0.4)
        plt.ylim(-3*p4 - 0.2, 3*p4 + 0.2)
        plt.text(-p2 - 0.15, 0, r"$i^0$", fontsize=18)
        plt.text(-p4 - 0.15, p4 + 0.05, r"$i^+$", fontsize=18)
        plt.text(-p4 - 0.15, -p4 - 0.15, r"$i^-$", fontsize=18)
        plt.text(p4 + 0.05, -2*p4, r"$R=0$", fontsize=18, rotation=90, verticalalignment='center')
        
        plt.text(-1.4, -1.2, r"R\'egimen Subcr\'itico", fontsize=14)
        plt.text(-1.4, -1.35, rf"$C_{{CMC}} = {data['C_val']:.2f}$", fontsize=14)
        plt.text(-1.4, -1.50, rf"$R_1 = {R1:.3f} \quad R_2 = {R2:.3f}$", fontsize=14)
    else:
        plt.xlim(-p4 - 0.4, p2 + 0.4)
        plt.ylim(-p4 - 0.2, 3*p4 + 0.2)
        label_text = r"R\'egimen Cr\'itico" if regimen == 'critico' else r"R\'egimen Supercr\'itico"
        plt.text(-0.9, 0.0, label_text, fontsize=14)
        plt.text(-0.9, -0.15, rf"$C_{{CMC}} = {data['C_val']:.2f}$", fontsize=14)

    plt.text(p2 + 0.05, 0, r"$i^0$", fontsize=18)
    plt.text(p4 + 0.05, p4 + 0.05, r"$i^+$", fontsize=18)
    plt.text(p4, -p4 - 0.15, r"$i^-$", fontsize=18)
    plt.text(3*p4/2 + 0.01, p4/2, r"$\scri^+$", fontsize=18)
    plt.text(3*p4/2, -p4/2 - 0.05, r"$\scri^-$", fontsize=18)
    plt.text(-p4 - 0.15, 2*p4, r"$R=0$", fontsize=18, rotation=90, verticalalignment='center')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.yticks([])
    plt.xticks([])
    plt.gca().set_aspect('equal')
    plt.show()