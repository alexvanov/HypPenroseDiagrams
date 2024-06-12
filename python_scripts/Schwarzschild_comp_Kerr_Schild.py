import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from autograd import grad
from Interpolated_functions_Scw import alpha_num, beta_num, chi_num, grr_num, gthth_num, aconf_num
from Compactified_functions import omega, aconf_final, aconf_c, alpha_b, beta_b, grr_b, chi_b, alpha_c, beta_c, grr_c, chi_c
#from Schwarzschild_comp import hpcomp
from scipy.optimize import brentq
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
Q = 0

def num_derivative(f, x, h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)

def hfuncdcomp(r):
    Kcmc = -1
    Ccmc = 3.114839
    return (Kcmc*r**3+3*Ccmc*aconf_final(r)**3)*(-aconf_final(r)+r*num_derivative(aconf_final,r))/(r*aconf_final(r)**3*(r-2*Mass*aconf_final(r))* 
            np.sqrt(9+(Kcmc**2*r**2)/aconf_final(r)**2+(6*(Ccmc*Kcmc-3*Mass)*aconf_final(r))/r+
            (9*Ccmc**2*aconf_final(r)**4)/r**4))

def dfkscomp(r):
    A = alpha_b(r)**2 - beta_b(r)**2*grr_b(r)/chi_b(r)
    return -((aconf_final(r)-r*num_derivative(aconf_final,r))/aconf_final(r)**2)*(omega(r)**2-A)/A

def dfkscompc(r):
    A = alpha_b(r)**2 - beta_b(r)**2*grr_b(r)/chi_b(r)
    return -((aconf_c(r)-r*num_derivative(aconf_c,r))/aconf_c(r)**2)*(omega(r)**2-A)/A
    return 

def dhpccomp(r):
    c = 1
    A = (alpha_c(r)**2-beta_c(r)**2*grr_c(r)/chi_c(r))/c**2
    return -beta_c(r)*grr_c(r)/chi_c(r)/c/A - dfkscompc(r)

'''
r1 = np.arange(0.0001, 0.9999, 0.0001)
plt.plot(r1, hfuncdcomp(r1)-dfkscomp(r1), color='blue')
plt.plot(r1, dhpccomp(r1), color='red', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-200,250)
plt.show()
'''

# Find rhornum (location of the horizon in the compactified radial coordinate)
def equation1(r):
    return r/aconf_c(r) - 2*Mass

roots = brentq(equation1,0.0001,0.9999) # Find the roots of eq1 between 0 and 1
rhornum = round(roots,6)
#print('rhornum =', rhornum)

def hfu_p(r, h):
    return np.array([dhpccomp(r)])

closehor = 0.000001

sol_1_back = solve_ivp(hfu_p, [0.8, 0+closehor], [0], t_eval=np.linspace(0.8, 0+closehor, 300000), method='Radau')
sol_1_forw = solve_ivp(hfu_p, [0.8, 1-closehor], [0], t_eval=np.linspace(0.8, 1-closehor, 300000), method='Radau')
combined_r_1 = np.hstack((sol_1_back.t[::-1], sol_1_forw.t[1:]))
combined_hfu_1 = np.hstack((sol_1_back.y[0][::-1], sol_1_forw.y[0][1:]))
def dhcnumout(r):
    return PchipInterpolator(combined_r_1, combined_hfu_1)(r)

sol_2_back = solve_ivp(hfu_p, [0.5, 0+closehor], [0], t_eval=np.linspace(0.5, 0+closehor, 300000), method='Radau')
sol_2_forw = solve_ivp(hfu_p, [0.5, 1-closehor], [0], t_eval=np.linspace(0.5, 1-closehor, 300000), method='Radau')
combined_r_2 = np.hstack((sol_2_back.t[::-1], sol_2_forw.t[1:]))
combined_hfu_2 = np.hstack((sol_2_back.y[0][::-1], sol_2_forw.y[0][1:]))
def dhcnumin(r):
    return PchipInterpolator(combined_r_2, combined_hfu_2)(r)

def dhcnumin_final(r, tvalindex):
    if tvalindex > 1: return dhcnumin(r)-dhcnumin(rhornum-10**(-8))+dhcnumout(rhornum)
    else: return dhcnumin(r)-dhcnumin(rhornum)+dhcnumout(rhornum)

def dhcnumcomp(r):
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r <= rhornum
    mask2 = r > rhornum
    results = np.empty_like(r)
    results[mask1] = dhcnumin_final(r[mask1],0)
    results[mask2] = dhcnumout(r[mask2])
    return results

'''
r2 = np.arange(0.001, 0.999, 0.000001)
plt.plot(r2, omega(r2)*dhcnumcomp(r2))
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-12,2)
plt.show()
'''

# Setting the integration constant for the height function so that they
# conincide with the Minkowski one asymptotically

rphysmax = 100000

def equation2(r):
    return rphysmax-r/aconf_c(r)

roots = brentq(equation2,0.0001,0.99999999) # Find the roots of eq1 between 0 and 1
rcompmax = round(roots,6)
#print(rcompmax)

def minkheight(r):
    Kcmc = -1
    return np.sqrt(3**2/Kcmc**2+r**2)+3/Kcmc

def fks(r):
    rphyshornum = 2
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r < rphyshornum
    mask2 = r > rphyshornum
    results = np.empty_like(r)
    results[mask1] = -2*np.log(1-r[mask1]/2)
    results[mask2] = -2*np.log(-1+r[mask2]/2)
    return results

hconstval = minkheight(rcompmax/omega(rcompmax))-dhcnumcomp(rcompmax)-fks(rcompmax/aconf_c(rcompmax))
#print(hconstval)

def dhccomp(r):
    return dhcnumcomp(r)+hconstval

# Check if the coincide at infinity
'''
r3 = np.arange(0.9, 1, 0.000001)
plt.plot(r3, (dhcnumcomp(r3)+fks(r3/aconf_c(r3)))/minkheight(r3/omega(r3)))
plt.plot(r3, np.ones_like(r3))
plt.show()
'''

# Calculate hccomp again for the plots
closehor2 = 0.001

def hpccomp(r):
    return -beta_c(r)*grr_c(r)/chi_c(r)/(alpha_c(r)**2-beta_c(r)**2 * grr_c(r)/chi_c(r))

def hf_p(r, h):
    return np.array([hpccomp(r)])

sol_1_back_2 = solve_ivp(hf_p, [rhornum-closehor2, 0.00003], [0], t_eval=np.linspace(rhornum-closehor2, 0.00003, 300000), method='Radau')
sol_1_forw_2 = solve_ivp(hf_p, [rhornum-closehor2, 0.99997], [0], t_eval=np.linspace(rhornum-closehor2, 0.99997, 300000), method='Radau')
combined_r_1_2 = np.hstack((sol_1_back_2.t[::-1], sol_1_forw_2.t[1:]))
combined_hfu_1_2 = np.hstack((sol_1_back_2.y[0][::-1], sol_1_forw_2.y[0][1:]))
def hfu1(r):
    return PchipInterpolator(combined_r_1_2, combined_hfu_1_2)(r)
    
sol_2_back_2 = solve_ivp(hf_p, [rhornum+closehor2, 0.00003], [0], t_eval=np.linspace(rhornum+closehor2, 0.00003, 300000), method='Radau')
sol_2_forw_2 = solve_ivp(hf_p, [rhornum+closehor2, 0.99997], [0], t_eval=np.linspace(rhornum+closehor2, 0.99997, 300000), method='Radau')
combined_r_2_2 = np.hstack((sol_2_back_2.t[::-1], sol_2_forw_2.t[1:]))
combined_hfu_2_2 = np.hstack((sol_2_back_2.y[0][::-1], sol_2_forw_2.y[0][1:]))
def hfu2(r):
    return PchipInterpolator(combined_r_2_2, combined_hfu_2_2)(r)

def hcnumcomp2(r):
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r <= rhornum
    mask2 = r > rhornum
    results = np.empty_like(r)
    results[mask1] = hfu1(r[mask1])
    results[mask2] = hfu2(r[mask2])
    return results

hconstval2 = minkheight(rcompmax/omega(rcompmax))-hcnumcomp2(rcompmax)

def hccomp(r):
    return hcnumcomp2(r)+hconstval2


# Plots
r = np.arange(0.0001, rcompmax, 0.0001)
plt.plot(r, omega(r)*minkheight(r/omega(r)), color='blue', label=r'$\Omega h$ \text{(Mink)}')
plt.plot(r, omega(r)*hccomp(r), color='green', label=r'$\Omega h$ \text{(Schw)}')
plt.plot(r, omega(r)*fks(r/aconf_c(r)), color='black', linestyle='--', label=r'$\Omega f$ \text{(Schw)}')
plt.plot(r, omega(r)*dhccomp(r), color='black', label=r'$\Omega \Delta h$ \text{(Schw)}')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-12,4)
plt.yticks([-10, -5, 0])
plt.xlabel(r'$r$')
plt.legend(frameon=False, loc='lower right', fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_comp_omega(h,h,f,dh),f,delta.pdf')
plt.savefig(file_path)
plt.show()

r = np.arange(0.0001, rcompmax, 0.0001)
plt.plot(r, omega(r)*minkheight(r/omega(r)), color='purple', linestyle='-.', label=r'$\Omega h$ \text{(Mink)}')
plt.plot(r,aconf_c(r)*hccomp(r), color='blue', label=r'$\overline{\Omega} h$ \text{(Schw)}')
plt.plot(r, omega(r)*fks(r/aconf_c(r)), color='black', linestyle='--', label=r'$\Omega f$ \text{(Schw)}')
plt.plot(r, aconf_c(r)*dhccomp(r), color='black', label=r'$\overline{\Omega} \Delta h$ \text{(Schw)}')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-3.2,3.2)
plt.xlabel(r'$r$')
plt.legend()
plt.legend(loc='lower right')
plt.show()

'Repeat the code writen in Schwarzschild_comp.py for the dashed orange'

# Find rhornum (location of the horizon in the compactified radial coordinate)
def equation1(r):
    return r/aconf_final(r) - 2*Mass

roots = brentq(equation1,0.0001,0.9999) # Find the roots of eq1 between 0 and 1
rhornum_2 = round(roots,6)
#print('rhornum =', rhornum)

# Equations --> Constant radius curves
getclose = 0.001
list_radius_out = [rhornum_2 + getclose, 0.15, 0.22, 0.3, 0.4, 0.6, 0.94]   #7
list_radius_in = [0 + getclose, 0.01, 0.05, 0.1, rhornum_2 - getclose]    #5
first_iteration1 = True
first_iteration2 = True
tphys = np.arange(-100, 100, 0.0001)
for r in list_radius_in:
    r1 = r/aconf_final(r)

    if r == list_radius_in[0]:
        tphys = np.arange(-100, 80, 0.0001)
        u_in = np.sqrt(1-r1/(2*Mass))*np.exp(r1/(4*Mass))*np.sinh(tphys/(4*Mass))
        v_in = np.sqrt(1-r1/(2*Mass))*np.exp(r1/(4*Mass))*np.cosh(tphys/(4*Mass))
        U_in = np.arctan(u_in + v_in)
        V_in = np.arctan(u_in - v_in)
        T_in = (U_in - V_in)/2
        R_in = (U_in + V_in)/2
        plt.plot(R_in, T_in, color='orange', linestyle='--', linewidth=2.7)

    else:
        u_in = np.sqrt(1-r1/(2*Mass))*np.exp(r1/(4*Mass))*np.sinh(tphys/(4*Mass))
        v_in = np.sqrt(1-r1/(2*Mass))*np.exp(r1/(4*Mass))*np.cosh(tphys/(4*Mass))

        U_in = np.arctan(u_in + v_in)
        V_in = np.arctan(u_in - v_in)

        T_in = (U_in - V_in)/2
        R_in = (U_in + V_in)/2
        plt.plot(R_in, T_in, color='orange', linestyle='--', linewidth=1.3)

for r in list_radius_out:
    r1= r/aconf_final(r)
    u_out = np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.cosh(tphys/(4*Mass))
    v_out = np.sqrt(r1/(2*Mass)-1)*np.exp(r1/(4*Mass))*np.sinh(tphys/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2

    valid_T = (T_out > -np.pi/4) & (T_out < np.pi/4)

    if first_iteration1:
        plt.plot(R_out[valid_T], T_out[valid_T], color='orange', linestyle='--', linewidth=1.3, label=r'$\tilde{r} \, \text{const}$')
        first_iteration1 = False
    else:
       plt.plot(R_out[valid_T], T_out[valid_T], color='orange', linestyle='--', linewidth=1.3)

'Now for the compactified radial coordinate'

first_iteration3 = True
getclose3 = 0.001

list_time1 = [39, 30, 26.25, 23.75, 21.5, 18]

def draw_s(t, first):
    rphys_brown = np.arange(0+getclose3, 1-0.06, 0.0001)
    t = t + dhccomp(rphys_brown)
    rphys_brown = rphys_brown/aconf_c(rphys_brown)

    u_s = (np.exp((t+rphys_brown)/(4*Mass))+(rphys_brown/(2*Mass)-1)*np.exp(-(t-rphys_brown)/(4*Mass)))/2
    v_s = (np.exp((t+rphys_brown)/(4*Mass))-(rphys_brown/(2*Mass)-1)*np.exp(-(t-rphys_brown)/(4*Mass)))/2
    
    U_s = np.arctan(u_s + v_s)
    V_s = np.arctan(u_s - v_s)

    T_s = (U_s - V_s)/2
    R_s = (U_s+ V_s)/2

    if first:
        plt.plot(R_s, T_s, color='brown', linewidth=1.3, label=r'$t = \text{const}$')
        first = False
    else:
       plt.plot(R_s, T_s, color='brown', linewidth=1.3)

    return first

for t in list_time1:
    first_iteration3 = draw_s(t, first_iteration3)

list_time2 = [20, 13, 10, 8, 5, 0]

# Create the diagram
plt.plot([0,(np.pi)/4], [0, (np.pi)/4], 'k-', linewidth=2)
plt.plot([0,(np.pi)/4], [0, -(np.pi)/4], 'k-', linewidth=2)
plt.plot([(np.pi)/4, (np.pi)/2], [-(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([(np.pi)/4, (np.pi)/2], [(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, 0], [(np.pi)/4, 0], 'k-', linewidth=2)
plt.plot([-(np.pi)/4, (np.pi)/4], [(np.pi)/4, (np.pi)/4], 'k-', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
for spine in plt.gca().spines.values():
    spine.set_visible(False)  # Hide spines
plt.yticks([])
plt.xticks([])
plt.xlim(-(np.pi)/4-0.1, (np.pi)/2+0.3)
plt.ylim(-(np.pi)/4-0.3, (np.pi)/4 +0.3)
plt.text((3*np.pi)/8+0.05, (np.pi)/8+0.05, r"$ \scri ^{+}$", fontsize = 18)
plt.text((3*np.pi)/8+0.05, -(np.pi)/8-0.15, r"$ \scri ^{-}$", fontsize = 18)
plt.text((np.pi)/2+0.05, -0.05, r"$ i ^{0}$", fontsize = 18)
plt.text((np.pi)/4, -(np.pi)/4-0.15, r"$ i ^{-}$", fontsize = 18)
plt.text((np.pi)/4, (np.pi)/4+0.05, r"$ i ^{+}$", fontsize = 18)
plt.text(-0.15, (np.pi)/4+0.05, r'$\tilde{r} = 0$', fontsize = 18)
plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.1, 0.2), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_comp_Kerr_Schild.pdf')
plt.savefig(file_path)
plt.show()

####################################3
