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

# Define the quantities
list_lightout = [-2.5, -1.1, -0.5, -0.1, 0.3, 0.8, 1.6, 4]
list_lightin = [-2.5, -1.1, -0.5, -0.1, 0.3, 0.8, 1.6, 4]

first_iteration1 = True #Used to create a label
first_iteration2 = True

u = np.arange(-50, 50, 0.001)
for v in list_lightin:
    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    # Filter data where R > 0
    filter = R > 0
    T_filtered = T[filter]
    R_filtered = R[filter]

    if first_iteration1:
        plt.plot(R_filtered, T_filtered, linestyle='dotted', color='blue', linewidth=1.3, label=r'$\Tilde{V}$ = const')
        first_iteration1 = False
    else:
       plt.plot(R_filtered, T_filtered, linestyle='dotted', color='blue', linewidth=1.3)

v1 = np.arange(-50, 50, 0.001)
for u1 in list_lightout:
    U = np.arctan(u1)
    V = np.arctan(v1)

    T = (V + U)/2
    R = (V - U)/2
    
    # Filter data where R > 0
    filter = R > 0
    T_filtered = T[filter]
    R_filtered = R[filter]

    if first_iteration2:
        plt.plot(R_filtered, T_filtered, color='blue', linewidth=1.3, label=r'$\Tilde{U}$ = const')
        first_iteration2 = False
    else:
        plt.plot(R_filtered, T_filtered, color='blue', linewidth=1.3)

# Create the diagram
plt.plot([0,(np.pi)/2], [(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,(np.pi)/2], [-(np.pi)/2, 0], 'k-', linewidth=2)
plt.plot([0,0], [-(np.pi)/2, (np.pi)/2], 'k-', linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
for spine in plt.gca().spines.values():
    spine.set_visible(False)  # Hide spines
plt.yticks([])
plt.xticks([])
plt.xlim(-0.2, (np.pi)/2+0.6)
plt.ylim(-(np.pi)/2-0.5, (np.pi)/2 +0.5)
plt.text((np.pi)/4+0.1, (np.pi)/4+0.1, r"$ \scri ^{+}$", fontsize = 18)
plt.text((np.pi)/4+0.1, -(np.pi)/4-0.3, r"$ \scri ^{-}$", fontsize = 18)
plt.text((np.pi)/2+0.1, -0.1, r"$ i ^{0}$", fontsize = 18)
plt.text(0, -(np.pi)/2-0.3, r"$ i ^{-}$", fontsize = 18)
plt.text(0, (np.pi)/2+0.1, r"$ i ^{+}$", fontsize = 18)
plt.legend(frameon=False, loc='lower right', fontsize=12)
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_in_out.pdf')
plt.savefig(file_path)
plt.show()