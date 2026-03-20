import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from Critical_Ccmc import Ccmc_crit
from Compactified_functions import aconf_final
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

# Set the value of the quantities
rscri = 1
Kcmc = -1
Q = 0
m = 1
Ccmc = Ccmc_crit(1)
time = [10, 3.5, 1.25, 0.25, -0.5, -2, -5]
r = np.arange(0.001, 0.999, 0.01)
first_iteration1 = True

def omega(r):
    rscri = 1
    return (-r**2+rscri**2)/(2*(-3/Kcmc)*rscri)

def num_derivative(f, x, h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)

def hpcompcmc(r):
    return ((aconf_final(r)-r*num_derivative(aconf_final,r))/aconf_final(r)**2)*np.sqrt((1-(np.cos(np.pi*r/rscri))**5)/2)
def h_prime1(r, h): # Useful since solve_ivp needs func(r, h)
    return hpcompcmc(r)

def hfcompcmc(r):
    r_span = (0.001, 0.9997) # Define the range of r values
    h0 = [0] # Initial value of h at r=0
    sol = solve_ivp(h_prime1, r_span, h0, t_eval=np.linspace(0.001, 0.9997, 5000), method='LSODA')
    r_values = sol.t
    h_values = sol.y[0]
    return interp1d(r_values, h_values, kind= 'cubic', fill_value="extrapolate")(r)

######################################################################

for t0 in time:
    u = t0 + hfcompcmc(r) - r/omega(r)
    v = t0 + hfcompcmc(r) + r/omega(r)

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    if first_iteration1:
        plt.plot(R,T, color='purple', linewidth=1.3, label=r'$h_{1}(r)$')
        first_iteration1 = False
    else: plt.plot(R,T, color='purple', linewidth=1.3)

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
plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(0.7, 0.1), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_hf_cos.pdf')
plt.savefig(file_path)
plt.show()


#############################################

# Set the value of the quantities
rscri = 1
Kcmc = -1
Ccmc = Ccmc_crit(1)
time = [10, 3.5, 1.25, 0.25, -0.5, -2, -5]
r = np.arange(0.001, 0.999, 0.01)
first_iteration1 = True

def omega(r):
    rscri = 1
    return (-r**2+rscri**2)/(2*(-3/Kcmc)*rscri)

def num_derivative(f, x, h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)

def hpcompcmc(r):
    return ((aconf_final(r)-r*num_derivative(aconf_final,r))/aconf_final(r)**2)*(r**2*(3*rscri**2-r**2))/(rscri**2*(rscri**2+r**2))

def h_prime1(r, h): # Useful since solve_ivp needs func(r, h)
    return hpcompcmc(r)

def hfcompcmc(r):
    r_span = (0.001, 0.9997) # Define the range of r values
    h0 = [0] # Initial value of h at r=0
    sol = solve_ivp(h_prime1, r_span, h0, t_eval=np.linspace(0.001, 0.9997, 5000), method='LSODA')
    r_values = sol.t
    h_values = sol.y[0]
    return interp1d(r_values, h_values, kind= 'cubic', fill_value="extrapolate")(r)

######################################################################

for t0 in time:
    u = t0 + hfcompcmc(r) - r/omega(r)
    v = t0 + hfcompcmc(r) + r/omega(r)

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    if first_iteration1:
        plt.plot(R,T, color='olive', linewidth=1.3, label=r'$h_{2}(r)$')
        first_iteration1 = False
    else: plt.plot(R,T, color='olive', linewidth=1.3)

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
plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(0.7, 0.1), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_hf_2.pdf')
plt.savefig(file_path)
plt.show()