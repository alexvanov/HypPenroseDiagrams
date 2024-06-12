import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Compactified_functions import *
from Interpolated_functions_Scw import aconf_num
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

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

def ReadPygraph(filename):
    times = []
    data = []

    # Read the file line by line
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#Time ='):
                time_value = float(line.split('=')[1].strip())
                times.append(time_value)
            elif line:
                data_values = [float(value) for value in line.split()]
                data.append(data_values)
    
    times = np.array(times)
    data = np.array(data).reshape((len(times), -1, 2))
    #print(f"read {data.nbytes / 1024:.2f} KByte of data from {filename};")
    
    return times, data

tagdiag = "imitateoldoslong"
num_files = 25-9
diagdatam = [None] * num_files

repolocation = "C:/Users/jacma/OneDrive - Universidade de Lisboa/Desktop/PIC/Penrose Diagrams/Alex/HypPenroseDiagrams-main/data/data/"

# Loop to read and process data
for num in range(num_files):
    filenamesdiag = [
        os.path.join(repolocation, tagdiag + "m" + str(num + 10) + "_diag_R_c4_single.yg"),
        os.path.join(repolocation, tagdiag + "m" + str(num + 10) + "_diag_T_c4_single.yg")
    ]
    #print(tagdiag + "m" + str(num+10))

    times_R, data_R = ReadPygraph(filenamesdiag[0])
    times_T, data_T = ReadPygraph(filenamesdiag[1])
    
    diagdatam[num] = (times_R, data_R, data_T)


fig, ax = plt.subplots()

# Plot the diagram
ax.plot([0, (np.pi)/4], [0, (np.pi)/4], 'k-', linewidth=2)
ax.plot([0, (np.pi)/4], [0, -(np.pi)/4], 'k-', linewidth=2)
ax.plot([(np.pi)/4, (np.pi)/2], [-(np.pi)/4, 0], 'k-', linewidth=2)
ax.plot([(np.pi)/4, (np.pi)/2], [(np.pi)/4, 0], 'k-', linewidth=2)
ax.plot([-(np.pi)/4, 0], [(np.pi)/4, 0], 'k-', linewidth=2)
ax.plot([-(np.pi)/4, (np.pi)/4], [(np.pi)/4, (np.pi)/4], 'k-', linewidth=2)
ax.set_aspect('equal', adjustable='box')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-(np.pi)/4 - 0.1, (np.pi)/2 + 0.3)
ax.set_ylim(-(np.pi)/4 - 0.3, (np.pi)/4 + 0.3)
ax.text((3*np.pi)/8 + 0.05, (np.pi)/8 + 0.05, r"$ \scri ^{+}$", fontsize=18)
ax.text((3*np.pi)/8 + 0.05, -(np.pi)/8 - 0.15, r"$ \scri ^{-}$", fontsize=18)
ax.text((np.pi)/2 + 0.05, -0.05, r"$ i ^{0}$", fontsize=18)
ax.text((np.pi)/4, -(np.pi)/4 - 0.15, r"$ i ^{-}$", fontsize=18)
ax.text((np.pi)/4, (np.pi)/4 + 0.05, r"$ i ^{+}$", fontsize=18)
ax.text(-0.15, (np.pi)/4 + 0.05, r'$\tilde{r} = 0$', fontsize=18)
ax.axis('off')

# Initialize the line objects for animation (one for each file)
lines = [ax.plot([], [], lw=2)[0] for _ in range(num_files)]

# Initialize function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    for i, line in enumerate(lines):
        times, data_R, data_T = diagdatam[i]
        x = data_R[frame, :, 1]  # Use the second column for x
        y = data_T[frame, :, 1]  # Use the second column for y
        line.set_data(x, y)
    return lines

anim = FuncAnimation(fig, animate, init_func=init, frames=len(diagdatam[0][0]), interval=200, blit=True)

# Save the animation as a gif
gif_path = "C:/Users/jacma/OneDrive - Universidade de Lisboa/Desktop/PIC/Gifs/trumrelax.gif"
anim.save(gif_path, writer=PillowWriter(fps=10))

plt.show()


###########################################################################

'Static plots'

def plot_penrose_diagram(ax):
    ax.plot([0, (np.pi)/4], [0, (np.pi)/4], 'k-', linewidth=2)
    ax.plot([0, (np.pi)/4], [0, -(np.pi)/4], 'k-', linewidth=2)
    ax.plot([(np.pi)/4, (np.pi)/2], [-(np.pi)/4, 0], 'k-', linewidth=2)
    ax.plot([(np.pi)/4, (np.pi)/2], [(np.pi)/4, 0], 'k-', linewidth=2)
    ax.plot([-(np.pi)/4, 0], [(np.pi)/4, 0], 'k-', linewidth=2)
    ax.plot([-(np.pi)/4, (np.pi)/4], [(np.pi)/4, (np.pi)/4], 'k-', linewidth=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-(np.pi)/4 - 0.1, (np.pi)/2 + 0.3)
    ax.set_ylim(-(np.pi)/4 - 0.3, (np.pi)/4 + 0.3)
    ax.text((3*np.pi)/8 + 0.05, (np.pi)/8 + 0.05, r"$ \scri ^{+}$", fontsize=18)
    ax.text((3*np.pi)/8 + 0.05, -(np.pi)/8 - 0.15, r"$ \scri ^{-}$", fontsize=18)
    ax.text((np.pi)/2 + 0.05, -0.05, r"$ i ^{0}$", fontsize=18)
    ax.text((np.pi)/4, -(np.pi)/4 - 0.15, r"$ i ^{-}$", fontsize=18)
    ax.text((np.pi)/4, (np.pi)/4 + 0.05, r"$ i ^{+}$", fontsize=18)
    ax.text(-0.15, (np.pi)/4 + 0.05, r'$\tilde{r} = 0$', fontsize=18)
    ax.axis('off')

# Function to plot specific frames
def plot_static_data(filename, shift, points):
    fig, ax = plt.subplots()
    
    for i in range(1, points + 1):
        time_index = i - 1
        file_index = i - 1 + shift

        if file_index < len(diagdatam) and time_index < len(diagdatam[file_index][2]):
            x = diagdatam[file_index][1][time_index][:, 1]  # Use the second column for x (R)
            y = diagdatam[file_index][2][time_index][:, 1]  # Use the second column for y (T)
            ax.plot(x, y)
        
    plot_penrose_diagram(ax)
    plt.savefig(filename)
    plt.show()

# Set up directories and paths
output_dir = "C:/Users/jacma/OneDrive - Universidade de Lisboa/Desktop/PIC/Static plots"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each set of data
plot_static_data(os.path.join(output_dir, "plot1.pdf"), 0, 8)   # Corresponds to [#, #, 2]
plot_static_data(os.path.join(output_dir, "plot2.pdf"), 3, 8)   # Corresponds to [# + 3, #, 2]
plot_static_data(os.path.join(output_dir, "plot3.pdf"), 6, 8)   # Corresponds to [# + 6, #, 2]
plot_static_data(os.path.join(output_dir, "plot4.pdf"), 9, 7)   # Corresponds to [# + 9, #, 2]


getclose = 0.001
Mass = 1

def plot_compar2cmc(filename):
    fig, ax = plt.subplots()
    first_iteration = True
    second_iteration = True

    indices = [(1, 1), (10, 10), (4, 1), (8, 5), (7, 1), (11, 5), (12, 1), (16, 5)]

    for (file, time) in indices:
        x = diagdatam[file - 1][1][time - 1][:, 1]  # Use the second column for x (R)
        y = diagdatam[file - 1][2][time - 1][:, 1]  # Use the second column for y (T)

        if first_iteration or second_iteration:
            if time == 1:
                ax.plot(x, y, color='blue', label=r'$\text{CMC}$')
                first_iteration = False

            else: 
                ax.plot(x, y, color='black', label=r'$\text{stationary}$')
                second_iteration = False

        else:
            if time == 1: ax.plot(x, y, color='blue')
            else: ax.plot(x, y, color='black')

    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 0.2), fontsize=12)
    plot_penrose_diagram(ax)

    tphys = np.arange(-100, 80, 0.001)
    listrsicmc = [0 + getclose]
    listrsinoncmc = [0 + getclose]

    for r in listrsicmc:
        r = r / aconf_final(r)
        u_in = np.sqrt(1 - r / (2 * Mass)) * np.exp(r / (4 * Mass)) * np.sinh(tphys / (4 * Mass))
        v_in = np.sqrt(1 - r / (2 * Mass)) * np.exp(r / (4 * Mass)) * np.cosh(tphys / (4 * Mass))

        U_in = np.arctan(u_in + v_in)
        V_in = np.arctan(u_in - v_in)

        T_in = (U_in - V_in) / 2
        R_in = (U_in + V_in) / 2

        ax.plot(R_in, T_in, linestyle='--', color='blue', linewidth=1.3, label=r'$\text{CMC trumpet}$')

    for r in listrsinoncmc:
        r = r / aconf_num(r)
        u_in = np.sqrt(1 - r / (2 * Mass)) * np.exp(r / (4 * Mass)) * np.sinh(tphys / (4 * Mass))
        v_in = np.sqrt(1 - r / (2 * Mass)) * np.exp(r / (4 * Mass)) * np.cosh(tphys / (4 * Mass))

        U_in = np.arctan(u_in + v_in)
        V_in = np.arctan(u_in - v_in)

        T_in = (U_in - V_in) / 2
        R_in = (U_in + V_in) / 2

        ax.plot(R_in, T_in, linestyle='--', color='black', linewidth=1.3, label=r'$\text{stationary trumpet}$')

    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=12)
    plt.savefig(filename)
    plt.show()

# Plot and save the specific lines
plot_compar2cmc(os.path.join(output_dir, "compar2cmc.pdf"))



###########################################################################

'Comparison between height-function calculated (using delta_h + f height function integration) and eikonal evolved non-CMC stationary data'

# Repeat the code to calculate dhccomp(r)

def num_derivative(f, x, h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)

def dfkscompc(r):
    A = alpha_b(r)**2 - beta_b(r)**2*grr_b(r)/chi_b(r)
    return -((aconf_c(r)-r*num_derivative(aconf_c,r))/aconf_c(r)**2)*(omega(r)**2-A)/A
    return 

def dhpccomp(r):
    c = 1
    A = (alpha_c(r)**2-beta_c(r)**2*grr_c(r)/chi_c(r))/c**2
    return -beta_c(r)*grr_c(r)/chi_c(r)/c/A - dfkscompc(r)

def equation1(r):
    return r/aconf_c(r) - 2*Mass

roots = brentq(equation1,0.0001,0.9999) # Find the roots of eq1 between 0 and 1
rhornum = round(roots,6)

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

rphysmax = 100000

def equation2(r):
    return rphysmax-r/aconf_c(r)

roots = brentq(equation2,0.0001,0.99999999) # Find the roots of eq1 between 0 and 1
rcompmax = round(roots,6)

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

def dhccomp(r):
    return dhcnumcomp(r)+hconstval

# Now construct R and T for the height function integrated using delta_h and express them in terms
# of the compactified radial coordinate for ease of comparison

ibrad = 0.001
obrad = 0.946
addextra = 0


