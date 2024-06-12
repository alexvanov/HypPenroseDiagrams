import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from Compactified_functions import alpha_M, beta_M, chi_M, grr_M, gthth_M, aconf_c
from Perturbed_functions import alpha_p, beta_p, chi_p, grr_p, gthth_p
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
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

# Define the quantities
time = [10, 3.5, 1.25, 0.25, -0.5, -2, -5]
rscri = 1
Kcmc = -1
r = np.arange(0.001, 0.999, 0.01)

# Define useful functions
def omega(r):
    return (-r**2+rscri**2)/(2*(-3/Kcmc)*rscri)

# Label
first_iteration1 = True 
first_iteration2 = True

# CMC
def hpcompcmc(r):
    return ((-beta_M(r)*grr_M(r))/chi_M(r))/((alpha_M(r)**2)-(beta_M(r)**2)*grr_M(r)/chi_M(r))

def h_prime1(r, h): # Useful since solve_ivp needs func(r, h)
    return hpcompcmc(r)

def hfcompcmc(r):
    r_span = (0.001, 0.999) # Define the range of r values
    h0 = [0] # Initial value of h at r=0
    sol = solve_ivp(h_prime1, r_span, h0, t_eval=np.linspace(0.001, 0.999, 5000), method='LSODA')
    r_values = sol.t
    h_values = sol.y[0]
    return interp1d(r_values, h_values, kind= 'cubic', fill_value="extrapolate")(r)


for t0 in time:
    u = t0 + hfcompcmc(r) - r/omega(r)
    v = t0 + hfcompcmc(r) + r/omega(r)

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    if first_iteration1:
        plt.plot(R,T, color='orange', linewidth=1.3, label='CMC Minkowski')
        first_iteration1 = False
    else: plt.plot(R,T, color='orange', linewidth=1.3)

# Perturbed
def hpnumcomp(r):
    return ((-beta_p(r)*grr_p(r))/chi_p(r))/((alpha_p(r)**2)-(beta_p(r)**2)*grr_p(r)/chi_p(r))

def h_prime2(r, h): # Useful since solve_ivp needs func(r, h)
    return hpnumcomp(r)

def hfnumcomp(r):
    r_span = (0.001, 0.999) # Define the range of r values
    h0 = [0] # Initial value of h at r=0
    sol = solve_ivp(h_prime2, r_span, h0, t_eval=np.linspace(0.001, 0.999, 50000), method='LSODA')
    r_values = sol.t
    h_values = sol.y[0]
    return interp1d(r_values, h_values, kind= 'cubic', fill_value="extrapolate")(r)

def hfcomp(r):
    hconst = -1.3 #integration constant added to the numerical h so that it coincides
                  #with the Minkowski case at infinity
    return hfnumcomp(r) + hconst

for tp in time:
    u = tp + hfcomp(r) - r/omega(r)
    v = tp + hfcomp(r) + r/omega(r)

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    if first_iteration2:
        #plt.plot(R,T,'k-', linewidth=1.3, label='perturbed')
        first_iteration2 = False
    #else: plt.plot(R,T,'k-', linewidth=1.3)

# Create the diagram
plt.plot([0,(np.pi)/2], [(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,(np.pi)/2], [-(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,0], [-(np.pi)/2, (np.pi)/2], 'k-', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
for spine in plt.gca().spines.values():
    spine.set_visible(False)  # Hide spines
plt.yticks([])
plt.xticks([])
plt.xlim(-0.2, (np.pi)/2+1)
plt.ylim(-(np.pi)/2-0.9, (np.pi)/2 +0.5)
plt.text((np.pi)/4+0.1, (np.pi)/4+0.1, r"$ \scri ^{+}$", fontsize = 18)
plt.text((np.pi)/4+0.1, -(np.pi)/4-0.3, r"$ \scri ^{-}$", fontsize = 18)
plt.text((np.pi)/2+0.1, -0.1, r"$ i ^{0}$", fontsize = 18)
plt.text(0, -(np.pi)/2-0.3, r"$ i ^{-}$", fontsize = 18)
plt.text(0, (np.pi)/2+0.1, r"$ i ^{+}$", fontsize = 18)
plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(0.1, 0), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_CMC_vs_perturbed.pdf')
plt.savefig(file_path)
plt.show()


# Other Kcmc values (extra)

Kcmc_list = [(-1,'orange'), (-3,'green'), (-6,'blue'), (-9,'purple')]
time2= [3.5, 1.25, 0.25, -0.5, -2]

def hminkcomp(r, Kcmc):
    return np.sqrt((3/Kcmc)**2+(r/omega(r))**2)+3/Kcmc

for (Kcmc,col) in Kcmc_list:
    first_iteration1 = True
    for t0 in time2:
        u = t0 + hminkcomp(r,Kcmc) - r/omega(r)
        v = t0 + hminkcomp(r,Kcmc) + r/omega(r)

        U = np.arctan(u)
        V = np.arctan(v)

        T = (V + U)/2
        R = (V - U)/2

        if first_iteration1:
            plt.plot(R,T, linewidth=1.3, color=col, label=f'$K_{{CMC}}={Kcmc}$')
            first_iteration1 = False
        else: plt.plot(R,T, linewidth=1.3, color=col)

# Create the diagram
plt.plot([0,(np.pi)/2], [(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,(np.pi)/2], [-(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,0], [-(np.pi)/2, (np.pi)/2], 'k-', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
for spine in plt.gca().spines.values():
    spine.set_visible(False)  # Hide spines
plt.yticks([])
plt.xticks([])
plt.xlim(-0.2, (np.pi)/2+1)
plt.ylim(-(np.pi)/2-0.9, (np.pi)/2 +0.5)
plt.text((np.pi)/4+0.1, (np.pi)/4+0.1, r"$ \scri ^{+}$", fontsize = 18)
plt.text((np.pi)/4+0.1, -(np.pi)/4-0.3, r"$ \scri ^{-}$", fontsize = 18)
plt.text((np.pi)/2+0.1, -0.1, r"$ i ^{0}$", fontsize = 18)
plt.text(0, -(np.pi)/2-0.3, r"$ i ^{-}$", fontsize = 18)
plt.text(0, (np.pi)/2+0.1, r"$ i ^{+}$", fontsize = 18)
plt.legend(frameon=False, loc='lower right', fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_various_Kcmc.pdf')
plt.savefig(file_path)
plt.show()
