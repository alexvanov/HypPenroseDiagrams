import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from Critical_Ccmc import Ccmc_crit
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

# Define the function aconf(r) and its derivative
r = sp.symbols('r')
aconf = sp.Function('aconf')(r)
daconf_dr = sp.diff(aconf, r)

# Differential equation
equation = (9 * r**4 * (aconf - r * daconf_dr)**2) / (
    9 * Ccmc**2 * aconf**6 + 9 * Q**2 * aconf**4 * r**2 +
    6 * (Ccmc * Kcmc - 3 * m) * aconf**3 * r**3 +
    9 * aconf**2 * r**4 + Kcmc**2 * r**6)

dydr_expr = sp.solve(sp.Eq(equation, 1), daconf_dr)[0] # Isolate the derivative
func = sp.lambdify((r, aconf), dydr_expr, 'numpy') # Convert symbolic expression into a function 

# ODE system for solve_ivp
def ode_system(r, y):
    return func(r, y)

aconf_init = [0]  # Boundary condition at r=1 (aconf(1)=0)
r_span = (1, 0.001)
solution = solve_ivp(ode_system, r_span, aconf_init, t_eval=np.linspace(1, 0.001, 500), 
                     method='BDF', atol=1e-10, rtol=1e-6)

def aconf_final(r):
    aconf = interp1d(solution.t[::-1], solution.y[0][::-1], kind='cubic', fill_value="extrapolate", bounds_error=False)
    return aconf(r)

# Define omega
def omega(r):
    rscri = 1
    Kcmc = -1
    return (-r**2+rscri**2)/(2*(-3/Kcmc)*rscri)

#trumpetsubsb

def chi_b(r):
    return (aconf_final(r)/omega(r))**2

def gthth_b(r):
    return np.ones_like(r)

def grr_b(r):
    return np.ones_like(r)

def alpha_b(r):
    return (1/(3*aconf_final(r)*r**2))*np.sqrt(9*Ccmc**2*aconf_final(r)**6+
            9*Q**2*aconf_final(r)**4*r**2+6*(Ccmc*Kcmc-3*m)*aconf_final(r)**3*r**3+
            9*aconf_final(r)**2*r**4+Kcmc**2*r**6)*omega(r)

def beta_b(r):
    return ((Ccmc*aconf_final(r)**3)/r**2) + (Kcmc*r/3)

# Closed form

def chi_c(r):
    return chi_b(r)

def grr_c(r):
    return np.ones_like(r)

def gthth_c(r):
    return 1/np.sqrt(grr_c(r))

def alpha_c(r):
    return alpha_b(r)

def beta_c(r):
    return beta_b(r)

def aconf_c(r):
    return np.sqrt(chi_c(r)/gthth_c(r))*omega(r)

'Check that aconf_c and aconf_final are equal'
'''
r = np.arange(0, 1, 0.001)
plt.plot(r,aconf_final(r),color='orange', linestyle='--', label='aconf_final(r)')
plt.plot(r,aconf_c(r),color='blue', linestyle=':', label='aconf_c(r)')
plt.legend()
plt.show()
'''

# Minkowski CMC vs Perturbed
# mass = 0, Ccmc = 0, aconf(r) = omega(r)

def chi_M(r):
    return np.ones_like(r)

def grr_M(r):
    return np.ones_like(r)

def gthth_M(r):
    return 1/np.sqrt(grr_M(r))

def alpha_M(r):
    m = 0
    Ccmc = 0
    return (1/(3*omega(r)*r**2))*np.sqrt(9*Ccmc**2*omega(r)**6+
            9*Q**2*omega(r)**4*r**2+6*(Ccmc*Kcmc-3*m)*omega(r)**3*r**3+
            9*omega(r)**2*r**4+Kcmc**2*r**6)*omega(r)

def beta_M(r):
    Ccmc = 0
    return ((Ccmc*omega(r)**3)/r**2) + (Kcmc*r/3)



################################################

'Check that aconf_final and omega coincide at r=1'
'''
r1 = np.arange(0, 1, 0.00001)
plt.plot(r1, omega(r1), color='green', label=r'$\Omega(r)$')
plt.plot(r1, aconf_final(r1), color='blue', label=r'$\bar{\Omega}(r)$')
plt.plot(r1, r1 / 1.91, color='pink', label=r'$\frac{r}{\Tilde{r}_{trum}}$')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim([0, 0.20])
plt.xlabel(r"$r$")
plt.yticks([0, 0.05, 0.10, 0.15])
plt.legend(frameon=False, loc='upper right', fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Aconf_vs_Omega.pdf')
plt.savefig(file_path)
plt.show()
'''