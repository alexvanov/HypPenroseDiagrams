import matplotlib.pyplot as plt
import numpy as np
import os

# Enable LaTeX for text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set preamble to use necessary packages
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{mathrsfs}
\DeclareSymbolFontAlphabet{\mathrsfs}{rsfs}
\newcommand{\scri}{\mathrsfs{I}}
\newcommand{\scripx}{$\scri^+$}
\newcommand{\textconst}{\text{const}}
'''

Mass = 1
list_time_out = [0, 1.6, 4, 50, -1.6, -4, -50]             #7
list_time_in = [0, 1.6, 4, 50, -1.6, -4, -50]              #7
list_radius_out = [2.001, 2.05, 2.2, 2.5, 3, 4, 6, 80]     #8
list_radius_in = [0, 1.4, 1.8, 1.95, 1.999]                #5

first_iteration1 = True #Used to create a label
first_iteration2 = True

tin = np.arange(-100, 100, 0.0001)
for r in list_radius_in:
    u_in = np.sqrt(1-r/(2*Mass))*np.exp(r/(4*Mass))*np.sinh(tin/(4*Mass))
    v_in = np.sqrt(1-r/(2*Mass))*np.exp(r/(4*Mass))*np.cosh(tin/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    if first_iteration1:
        plt.plot(R_in,T_in,'b--', linewidth=1.3, label=r'$\tilde{r} = \text{const}$')
        first_iteration1 = False
    else:
       plt.plot(R_in,T_in,'b--', linewidth=1.3)

for r in list_radius_in:
    u_in = -np.sqrt(1-r/(2*Mass))*np.exp(r/(4*Mass))*np.sinh(tin/(4*Mass))
    v_in = -np.sqrt(1-r/(2*Mass))*np.exp(r/(4*Mass))*np.cosh(tin/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    plt.plot(R_in,T_in,'b--', linewidth=1.3)

rin = np.arange(0, 2.00000000001, 0.0001)
for t in list_time_in:
    u_in = np.sqrt(1-rin/(2*Mass))*np.exp(rin/(4*Mass))*np.sinh(t/(4*Mass))
    v_in = np.sqrt(1-rin/(2*Mass))*np.exp(rin/(4*Mass))*np.cosh(t/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    if first_iteration2:
        plt.plot(R_in,T_in,'b-', linewidth=1.3, label=r'$\tilde{t} = \text{const}$')
        first_iteration2 = False
    else:
       plt.plot(R_in,T_in,'b-', linewidth=1.3)

for t in list_time_in:
    u_in = -np.sqrt(1-rin/(2*Mass))*np.exp(rin/(4*Mass))*np.sinh(t/(4*Mass))
    v_in = -np.sqrt(1-rin/(2*Mass))*np.exp(rin/(4*Mass))*np.cosh(t/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    plt.plot(R_in,T_in,'b-', linewidth=1.3)



tout = np.arange(-100, 100, 0.0001)
for r1 in list_radius_out:
    u_out = np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.cosh(tout/(4*Mass))
    v_out = np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.sinh(tout/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2
    valid_T = (T_out > -np.pi/4) & (T_out < np.pi/4)

    plt.plot(R_out[valid_T],T_out[valid_T],'b--', linewidth=1.3)

for r1 in list_radius_out:
    u_out = -np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.cosh(tout/(4*Mass))
    v_out = -np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.sinh(tout/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2
    valid_T = (T_out > -np.pi/4) & (T_out < np.pi/4)

    plt.plot(R_out[valid_T],T_out[valid_T],'b--', linewidth=1.3)

rout = np.arange(2, 100, 0.0001)
for t1 in list_time_out:
    u_out = np.sqrt(rout/(2*Mass)-1)*np.exp(rout/(4*Mass))*np.cosh(t1/(4*Mass))
    v_out = np.sqrt(rout/(2*Mass)-1)*np.exp(rout/(4*Mass))*np.sinh(t1/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2

    plt.plot(R_out,T_out,'b-', linewidth=1.3)

for t1 in list_time_out:
    u_out = -np.sqrt(rout/(2*Mass)-1)*np.exp(rout/(4*Mass))*np.cosh(t1/(4*Mass))
    v_out = -np.sqrt(rout/(2*Mass)-1)*np.exp(rout/(4*Mass))*np.sinh(t1/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2

    plt.plot(R_out,T_out,'b-', linewidth=1.3)


# Create the diagram
plt.plot([0,(np.pi)/4], [0, (np.pi)/4], 'k-', linewidth=2)
plt.plot([0,(np.pi)/4], [0, -(np.pi)/4], 'k-', linewidth=2)
plt.plot([(np.pi)/4, (np.pi)/2], [-(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([(np.pi)/4, (np.pi)/2], [(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, 0], [(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, (np.pi)/4], [(np.pi)/4, (np.pi)/4], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, (np.pi)/4], [-(np.pi)/4, -(np.pi)/4], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, (np.pi)/4], [-(np.pi)/4, (np.pi)/4], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, -(np.pi)/2], [-(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, -(np.pi)/2], [(np.pi)/4, 0], 'k-', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
for spine in plt.gca().spines.values():
    spine.set_visible(False)  # Hide spines
plt.yticks([])
plt.xticks([])
plt.xlim(-(np.pi)/2-0.5, (np.pi)/2+0.1)
plt.ylim(-(np.pi)/4-0.7, (np.pi)/4 +0.1)
plt.text((3*np.pi)/8+0.05, (np.pi)/8+0.05, r"$ \scri ^{+}$", fontsize = 18)
plt.text(-(3*np.pi)/8-0.30, (np.pi)/8+0.05, r"$ \scri ^{+}$", fontsize = 18)
plt.text((3*np.pi)/8+0.05, -(np.pi)/8-0.15, r"$ \scri ^{-}$", fontsize = 18)
plt.text(-(3*np.pi)/8-0.30, -(np.pi)/8-0.15, r"$ \scri ^{-}$", fontsize = 18)
plt.text((np.pi)/2+0.05, -0.05, r"$ i ^{0}$", fontsize = 18)
plt.text(-(np.pi)/2-0.15, -0.05, r"$ i ^{0}$", fontsize = 18)
plt.text((np.pi)/4, -(np.pi)/4-0.20, r"$ i ^{-}$", fontsize = 18)
plt.text(-(np.pi)/4-0.03, -(np.pi)/4-0.20, r"$ i ^{-}$", fontsize = 18)
plt.text((np.pi)/4, (np.pi)/4+0.05, r"$ i ^{+}$", fontsize = 18)
plt.text(-(np.pi)/4-0.03, (np.pi)/4+0.05, r"$ i ^{+}$", fontsize = 18)
plt.text(-0.15, (np.pi)/4+0.05, r'$\tilde{r} = 0$', fontsize = 18)
plt.text(-0.15, -(np.pi)/4-0.20, r'$\tilde{r} = 0$', fontsize = 18)
plt.legend()
plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.1, 0), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_full_domain.pdf')
plt.savefig(file_path)
plt.show()