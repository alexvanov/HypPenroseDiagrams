import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from autograd import grad
from Interpolated_functions_Scw import alpha_num, beta_num, chi_num, grr_num, gthth_num, aconf_num
from Compactified_functions import omega, aconf_final, aconf_c, chi_c, alpha_c, beta_c
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

first_iteration1 = True #Used to create a label
first_iteration2 = True
first_iteration3 = True

'Note that aconf_c = aconf_final, so either one can be used.'

# Find rhornum (location of the horizon in the compactified radial coordinate)
def equation1(r):
    return r/aconf_final(r) - 2*Mass

roots = brentq(equation1,0.0001,0.9999) # Find the roots of eq1 between 0 and 1
rhornum = round(roots,6)
#print('rhornum =', rhornum)

# Equations --> Constant radius curves
getclose = 0.001
list_radius_out = [rhornum + getclose, 0.15, 0.22, 0.3, 0.4, 0.6, 0.94]   #7
list_radius_in = [0 + getclose, 0.01, 0.05, 0.1, rhornum - getclose]    #5

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
        plt.plot(R_in, T_in, color='orange', linestyle='--', linewidth=3, label=r'$\Tilde{r}_{trum}$')

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
        plt.plot(R_out[valid_T], T_out[valid_T], color='orange', linestyle='--', linewidth=1.3, label=r'$\tilde{r} = \text{const}$')
        first_iteration1 = False
    else:
       plt.plot(R_out[valid_T], T_out[valid_T], color='orange', linestyle='--', linewidth=1.3)

# Find the value for r_trumpet
'''
r_test1 = np.arange(0, 0.1, 0.001)
plt.plot(r_test1, r_test1/aconf_final(r_test1))
plt.xlim([0, 0.1])
plt.ylim([1.65, 1.95])
plt.show()
'''
r_trumpet = 0.01/aconf_final(0.01)
#print('r_trumpet =', r_trumpet)

# Construction using the physical radial coordinate (only for closed-from expressions)

def grphysrphysc(r):
    aconf = 1
    aconf_prime = 0
    r = r/aconf
    return chi_c(r)/alpha_c(r)**2 * (omega(r)**2/aconf**2 * (aconf-r*aconf_prime))**2

def beta_rphysc(r):
    aconf = 1
    aconf_prime = 0
    r = r/aconf
    return -np.sqrt((alpha_c(r)**2-omega(r)**2*(1-2*Mass*aconf/r +
            Q**2*aconf**2/r**2))*chi_c(r)/(chi_c(r)/alpha_c(r)**2 * (omega(r)**2/aconf**2 * (aconf- 
            r*aconf_prime))**2))


def hppcomp(r):
    return -beta_rphysc(r)*grphysrphysc(r)/chi_c(r)/(alpha_c(r)**2
            -beta_rphysc(r)**2*grphysrphysc(r)/chi_c(r))

def hfuncdcomp(r):
    Kcmc = -1
    Ccmc = 3.114839
    aconf = 1
    daconf = 0
    return (Kcmc*r**3+3*Ccmc*aconf**3)*(-aconf+r*daconf)/(r*aconf**3*(r-2*Mass*aconf)* 
            np.sqrt(9+(Kcmc**2*r**2)/aconf**2+(6*(Ccmc*Kcmc-3*Mass)*aconf)/r+
            (9*Ccmc**2*aconf**4)/r**4))

'''
r_test2 = np.arange(0.01, 10, 0.01)
plt.plot(r_test2,hppcomp(r_test2))
plt.plot(r_test2,np.zeros_like(r_test2))
plt.xlim([1,3])
plt.ylim([-20,150])
plt.show()
'''

'Find the expression for hfu'
rphyshornum = 2*Mass # Location of the horizon
rphystrum = r_trumpet # Location of the trumpet
rphysmax = 100000

def hf_p(r, h):
    #return hfuncdcomp(r)
    return np.array([hfuncdcomp(r)])  #CHANGE

sol_1_back = solve_ivp(hf_p, [rphyshornum-0.002, rphystrum+0.0002], [40], t_eval=np.linspace(rphyshornum-0.002, rphystrum+0.0002, 30000000), method='Radau')
sol_1_forw = solve_ivp(hf_p, [rphyshornum-0.002, rphyshornum-0.00000001], [40], t_eval=np.linspace(rphyshornum-0.002, rphyshornum-0.00000001, 30000000), method='Radau')
combined_r_1 = np.hstack((sol_1_back.t[::-1], sol_1_forw.t[1:]))
combined_hfu_1 = np.hstack((sol_1_back.y[0][::-1], sol_1_forw.y[0][1:]))
def hfu1(r):
    return PchipInterpolator(combined_r_1, combined_hfu_1)(r)
    #return interp1d(combined_r_1, combined_hfu_1, kind= 'linear', fill_value="extrapolate")(r)

sol_2_back = solve_ivp(hf_p, [rphyshornum+0.002, rphyshornum+0.00000001], [40], t_eval=np.linspace(rphyshornum+0.002, rphyshornum+0.00000001, 30000000), method='Radau')
sol_2_forw = solve_ivp(hf_p, [rphyshornum+0.002, rphysmax], [40], t_eval=np.linspace(rphyshornum+0.002, rphysmax, 30000000), method='Radau')
combined_r_2 = np.hstack((sol_2_back.t[::-1], sol_2_forw.t[1:]))
combined_hfu_2 = np.hstack((sol_2_back.y[0][::-1], sol_2_forw.y[0][1:]))
def hfu2(r):
    return PchipInterpolator(combined_r_2, combined_hfu_2)(r)
    #return interp1d(combined_r_2, combined_hfu_2, kind= 'linear', fill_value="extrapolate")(r)

def hpnumcomp(r):
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r <= rphyshornum
    mask2 = r > rphyshornum
    results = np.empty_like(r)
    results[mask1] = hfu1(r[mask1])
    results[mask2] = hfu2(r[mask2])
    return results

'''
r_test3 = np.arange(rphystrum, 5, 0.0000001)
plt.plot(r_test3, hpnumcomp(r_test3))
plt.show()
'''

# Setting the integration constant for the height function
# so that they conincide  with  the  Minkowski  one  asymptotically

def minkheight(r):
    Kcmc = -1
    return np.sqrt(3**2/Kcmc**2+r**2)+3/Kcmc

hconstval = (minkheight(100000)-hpnumcomp(100000))
#print(hconstval)

def hpcomp(r):
    return hpnumcomp(r)+hconstval

# Compare Mink vs Hfun
'''
r_test4 = np.arange(0.001, 1, 0.0000001)
plt.plot(r_test4, omega(r_test4)*minkheight(r_test4/omega(r_test4)), label='Mink')
plt.plot(r_test4, omega(r_test4)*hpcomp(r_test4/aconf_final(r_test4)), label='hf')
plt.xlim(0,1)
plt.ylim(-9,2)
plt.legend()
plt.show()
'''

# Plot the final height function
'''
rphys = np.arange(rphystrum, 5, 0.00001)
plt.plot(rphys, hpcomp(rphys), color='orange', linewidth=2)
plt.ylim(-30,-12)
plt.yticks([-25,-20,-15])
plt.xlabel(r"$\tilde{r}$")  # LaTeX formatting for the x-axis label
plt.ylabel(r'$h(\tilde{r})$')  # Label for the y-axis
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_comp_hf.pdf')
plt.savefig(file_path)
plt.show()
'''

# Equations --> Constant radius curves
getclose = 0.0001
list_time = [39, 30, 26.25, 23.75, 21.5, 18]

rphys_in = np.arange(rphystrum+getclose, rphyshornum-getclose, 0.00001)
for t in list_time:

    t = t + hpcomp(rphys_in)

    u_in = np.sqrt(1-rphys_in/(2*Mass))*np.exp(rphys_in/(4*Mass))*np.sinh(t/(4*Mass))
    v_in = np.sqrt(1-rphys_in/(2*Mass))*np.exp(rphys_in/(4*Mass))*np.cosh(t/(4*Mass))

    U_in = np.arctan(u_in + v_in)
    V_in = np.arctan(u_in - v_in)

    T_in = (U_in - V_in)/2
    R_in = (U_in + V_in)/2

    if first_iteration2:
        plt.plot(R_in, T_in, color='orange', linewidth=1.3, label=r'$t = \text{const}$ for $h$')
        first_iteration2 = False
    else:
       plt.plot(R_in, T_in, color='orange', linewidth=1.3)

rphys_out = np.arange(rphyshornum, 20, 0.00001)
for t in list_time:
    
    t = t + hpcomp(rphys_out)

    u_out = np.sqrt(rphys_out/(2*Mass)-1)*np.exp(rphys_out/(4*Mass))*np.cosh(t/(4*Mass))
    v_out = np.sqrt(rphys_out/(2*Mass)-1)*np.exp(rphys_out/(4*Mass))*np.sinh(t/(4*Mass))

    U_out = np.arctan(u_out + v_out)
    V_out = np.arctan(u_out - v_out)

    T_out = (U_out - V_out)/2
    R_out = (U_out + V_out)/2

    valid_T = (T_out > -np.pi/4) & (T_out < np.pi/4)

    plt.plot(R_out[valid_T], T_out[valid_T], color='orange', linewidth=1.3)


'Construction integrating for Delta_h = h - f  in physical Kerr-Schild coordinates (only for closed-form expressions)'

def fks(r):
    r = np.asarray(r)  # Ensure input is an array
    mask1 = r < rphyshornum
    mask2 = r > rphyshornum
    results = np.empty_like(r)
    results[mask1] = -2*np.log(1-r[mask1]/2)
    results[mask2] = -2*np.log(-1+r[mask2]/2)
    return results

def numerical_derivative(f, r, h=1e-7):
    # Check if the input is a scalar
    scalar_input = np.isscalar(r)
    r = np.atleast_1d(r)  # Ensure r is an array for the computations
    df = np.zeros_like(r)

    for i in range(len(r)):
        if r[i] < rphyshornum-h or r[i] > rphyshornum+h:
            df[i] = (f(r[i]+h)-f(r[i]-h))/(2*h)  # Central derivative
        elif r[i] <= rphyshornum:
            df[i] = (f(r[i])-f(r[i]-h))/h  # Left-hand derivative
        else:
            df[i] = (f(r[i]+h)-f(r[i]))/h  # Right-hand derivative

    return df if not scalar_input else df[0]

def hfu_p(r, h):
    return np.array([hfuncdcomp(r)-numerical_derivative(fks,r)])

sol_back = solve_ivp(hfu_p, [rphyshornum-0.002, rphystrum+0.0002], [40], t_eval=np.linspace(rphyshornum-0.002, rphystrum+0.0002, 30000000), method='LSODA')
sol_forw = solve_ivp(hfu_p, [rphyshornum-0.002, rphysmax], [40], t_eval=np.linspace(rphyshornum-0.002, rphysmax, 30000000), method='LSODA')
combined_r = np.hstack((sol_back.t[::-1], sol_forw.t[1:]))
combined_hfu = np.hstack((sol_back.y[0][::-1], sol_forw.y[0][1:]))

def dhpnumcomp(r):  # Kerr-Schild coordinates
    return PchipInterpolator(combined_r, combined_hfu)(r)
    #return interp1d(combined_r, combined_hfu, kind= 'linear', fill_value="extrapolate")(r)

hconstval_2 = (minkheight(rphysmax)-dhpnumcomp(rphysmax)-fks(rphysmax))

def dhpcomp(r):
    return dhpnumcomp(r)+hconstval_2


# Compare Mink vs dhpcomp + rks to see if the coincide at infinity
'''
r_test5 = np.arange(rphystrum+0.001, rphysmax, 0.01)
plt.plot(r_test5, minkheight(r_test5)/r_test5, label='Mink')
plt.plot(r_test5, (dhpcomp(r_test5)+fks(r_test5))/r_test5, label='dhp')
plt.ylim(0.9994,1)
plt.legend()
plt.show()
'''

# comparison between omega rescaled Minkowski (blue) and Schwarzschild (yellow intefrating for h and green integrating for
# delta_h) height functions and delta_h in red
'''
r6 = np.arange(0.001, 0.999, 0.000001)
plt.plot(r6, omega(r6)*minkheight(r6/omega(r6)), color='blue')
plt.plot(r6, omega(r6)*hpcomp(r6/aconf_final(r6)), color='orange')
plt.plot(r6, omega(r6)*(dhpcomp(r6/aconf_final(r6))+fks(r6/aconf_final(r6))), color='green')
plt.plot(r6, omega(r6)*dhpcomp(r6/aconf_final(r6)), color='red')
plt.xlim(0,1)
plt.ylim(-12,2)
plt.show()
'''

# h(r), f(r), delta_h(r) 
'''
r7 = np.arange(rphystrum+0.001, 5, 0.000001)
r7_2 = np.arange(0, 5, 0.001)
plt.plot(r7, hpcomp(r7)+12, color='orange', label=r'$h(\tilde{r})$')
plt.plot(r7, fks(r7), color='black', linestyle='--', label=r'$f(\tilde{r})$')
plt.plot(r7, dhpcomp(r7)+12, color='black', label=r'$\Delta h(\tilde{r})$')
plt.axhline(0, color='black', linewidth=0.5)
plt.ylim(-30,20)
plt.legend()
plt.legend(loc='lower right')
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_comp_h,f,delta.pdf')
plt.savefig(file_path)
plt.show()
'''

list_time2 = [39, 30, 26.25, 21.5]
getclose = 0.001

rphys_black = np.arange(rphystrum+getclose, 10, 0.00001)
for t in list_time2:

    t = t + dhpcomp(rphys_black)

    u_s = (np.exp((t+rphys_black)/(4*Mass))+(rphys_black/(2*Mass)-1)*np.exp(-(t-rphys_black)/(4*Mass)))/2
    v_s = (np.exp((t+rphys_black)/(4*Mass))-(rphys_black/(2*Mass)-1)*np.exp(-(t-rphys_black)/(4*Mass)))/2
    
    U_s = np.arctan(u_s + v_s)
    V_s = np.arctan(u_s - v_s)

    T_s = (U_s - V_s)/2
    R_s = (U_s+ V_s)/2

    if first_iteration3:
        plt.plot(R_s, T_s, color='black', linewidth=1.3, label=r'$t = \text{const}$ for $\Delta h$')
        first_iteration3 = False
    else:
       plt.plot(R_s, T_s, color='black', linewidth=1.3)

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
plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.2), fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Schwarzschild_phys_Kerr_Schild.pdf')
plt.savefig(file_path)
plt.show()