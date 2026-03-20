import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from autograd import grad
from Interpolated_functions_Scw import alpha_num, beta_num, chi_num, grr_num, gthth_num, aconf_num
from Compactified_functions import omega, aconf_final, aconf_c, chi_c, alpha_c, beta_c, grr_c
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

def hpccomp(r):
    return -beta_c(r)*grr_c(r)/chi_c(r)/(alpha_c(r)**2-beta_c(r)**2 * grr_c(r)/chi_c(r))

'''
r1 = np.arange(0.001, 1, 0.0001)
plt.plot(r1, hpccomp(r1))
plt.ylim(-200,250)
plt.show()
'''

# Find rhornum (location of the horizon in the compactified radial coordinate)
def equation1(r):
    return r/aconf_c(r) - 2*Mass

roots = brentq(equation1,0.0001,0.9999) # Find the roots of eq1 between 0 and 1
rhornum = round(roots,6)
#print('rhornum =', rhornum)
rphysmax = 100000

closehor = 0.001

def hf_p(r, h):
    return np.array([hpccomp(r)])

sol_1_back = solve_ivp(hf_p, [rhornum-closehor, 0.00003], [0], t_eval=np.linspace(rhornum-closehor, 0.00003, 300000), method='Radau')
sol_1_forw = solve_ivp(hf_p, [rhornum-closehor, 0.99997], [0], t_eval=np.linspace(rhornum-closehor, 0.99997, 300000), method='Radau')
combined_r_1 = np.hstack((sol_1_back.t[::-1], sol_1_forw.t[1:]))
combined_hfu_1 = np.hstack((sol_1_back.y[0][::-1], sol_1_forw.y[0][1:]))
def hfu1(r):
    return PchipInterpolator(combined_r_1, combined_hfu_1)(r)
    

sol_2_back = solve_ivp(hf_p, [rhornum+closehor, 0.00003], [0], t_eval=np.linspace(rhornum+closehor, 0.00003, 300000), method='Radau')
sol_2_forw = solve_ivp(hf_p, [rhornum+closehor, 0.99997], [0], t_eval=np.linspace(rhornum+closehor, 0.99997, 300000), method='Radau')
combined_r_2 = np.hstack((sol_2_back.t[::-1], sol_2_forw.t[1:]))
combined_hfu_2 = np.hstack((sol_2_back.y[0][::-1], sol_2_forw.y[0][1:]))
def hfu2(r):
    return PchipInterpolator(combined_r_2, combined_hfu_2)(r)

def hcnumcomp(r):
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r <= rhornum
    mask2 = r > rhornum
    results = np.empty_like(r)
    results[mask1] = hfu1(r[mask1])
    results[mask2] = hfu2(r[mask2])
    return results

'''
r2 = np.arange(0.00003, 0.99997, 0.000001)
plt.plot(r2, omega(r2)*hcnumcomp(r2))
plt.ylim(-8,5)
plt.show()
'''

# Setting the integration constant for the height function so that they
# conincide with the Minkowski one asymptotically -- using same
# condition as when integrating with the physical radial coordinate above

def equation2(r):
    return rphysmax-r/aconf_c(r)

roots = brentq(equation2,0.0001,0.99999999) # Find the roots of eq1 between 0 and 1
rcompmax = round(roots,6)
#print(rcompmax)

def minkheight(r):
    Kcmc = -1
    return np.sqrt(3**2/Kcmc**2+r**2)+3/Kcmc

hconstval = minkheight(rcompmax/omega(rcompmax))-hcnumcomp(rcompmax)
#print(hconstval)

def hccomp(r):
    return hcnumcomp(r)+hconstval

'''
r3 = np.arange(0.9, rcompmax, 0.000001)
plt.plot(r3, hccomp(r3)/minkheight(r3/omega(r3)))
plt.plot(r3, np.ones_like(r3))
plt.show()
'''

# Comparison between omega rescaled Minkowski (blue) and
# Schwarzschild (green) height functions (optionally also with the
# height function calculated using the physical radius (yellow))

'''
r4 = np.arange(0.00003, rcompmax, 0.0001)
plt.plot(omega(r4)*minkheight(r4/omega(r4)), color='blue')
plt.plot(omega(r4)*hccomp(r4), color='green')
plt.plot(omega(r4)*hpcomp(r4/aconf_final(r4)), color='orange')
plt.ylim(-9,2)
plt.show()
'''

r5 = np.arange(0.00003, rcompmax, 0.0001)
plt.plot(r5, minkheight(r5/omega(r5)), label='Minkowski', color='purple', linestyle='-.')
plt.plot(r5, hccomp(r5), label='Schwarzschild', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim()
plt.ylim(-70,60)
plt.xlabel(r'$r$')
plt.ylabel(r'$h(r)$')
plt.legend()
plt.legend(loc='lower right')
plt.show()


plt.plot(r5, omega(r5)*minkheight(r5/omega(r5)), label='Minkowski', color='purple', linestyle='-.')
plt.plot(r5, omega(r5)*hccomp(r5), label='Schwarzschild', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-9,1.2)
plt.xlabel(r'$r$')
plt.ylabel(r'$\Omega (r)h(r)$')
plt.legend()
plt.legend(loc='lower right')
plt.show()

first_iteration1 = True #Used to create a label
first_iteration2 = True

'Repeat the code writen in Schwarzschild_comp.py for the dashed orange lines'

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

getclose = 0.0005
list_time = [39, 30, 26.25, 23.75, 21.5, 18]

'Did it this way since the other method was resulting in overflows in sinh,cosh,exp'

def draw_in(t, first):
    r_comp_in = np.arange(0+getclose, rhornum-getclose, 0.00001)
    t = t + hccomp(r_comp_in)
    r_comp_in = r_comp_in/aconf_final(r_comp_in)
    
    u_in = np.sqrt(1-r_comp_in/(2*Mass))*np.exp(r_comp_in/(4*Mass))*np.sinh(t/(4*Mass))
    v_in = np.sqrt(1-r_comp_in/(2*Mass))*np.exp(r_comp_in/(4*Mass))*np.cosh(t/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    if first:
        plt.plot(R_in, T_in, color='red', linewidth=1.3, label=r'$t = \text{const}$ for $h$')
        first = False
    else:
        plt.plot(R_in, T_in, color='red', linewidth=1.3)
    
    return first

def draw_out(t):
    r_comp_out = np.arange(rhornum+getclose, 0.94, 0.00001)    
    t = t + hccomp(r_comp_out)
    r_comp_out = r_comp_out/aconf_final(r_comp_out)

    u_out = np.sqrt(r_comp_out/(2*Mass)-1)*np.exp(r_comp_out/(4*Mass))*np.cosh(t/(4*Mass))
    v_out = np.sqrt(r_comp_out/(2*Mass)-1)*np.exp(r_comp_out/(4*Mass))*np.sinh(t/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2

    valid_T = (T_out > -np.pi/4) & (T_out < np.pi/4)

    plt.plot(R_out[valid_T], T_out[valid_T], color='red', linewidth=1.3)

for t in list_time:
    first_iteration2 = draw_in(t, first_iteration2)
    draw_out(t)

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
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_comp_rad_coord.pdf')
plt.savefig(file_path)
plt.show()