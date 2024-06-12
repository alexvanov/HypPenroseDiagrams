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
list_time = [0, 0.3, 0.8, 1.6, 4, 15, -0.3, -0.8, -1.6, -4, -15]
list_radius = [0, 0.3, 0.8, 1.6, 4, 15]

first_iteration1 = True #Used to create a label
first_iteration2 = True

r1 = np.arange(0, 50, 0.001)
for t in list_time:
    u = t - r1
    v = t + r1

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2

    if first_iteration1:
        plt.plot(R,T,color='green', linewidth=1.3, label=r'$\tilde{t} = \text{const}$')
        first_iteration1 = False
    else: plt.plot(R,T,color='green', linewidth=1.3)

t1 = np.arange(-50, 50, 0.001)
for r in list_radius:
    u = t1 - r
    v = t1 + r

    U = np.arctan(u)
    V = np.arctan(v)

    T = (V + U)/2
    R = (V - U)/2
    
    if first_iteration2:
        plt.plot(R,T,color='green', linestyle='--', linewidth=1.3, label=r'$\tilde{r} = \text{const}$')
        first_iteration2 = False
    else: plt.plot(R,T,color='green', linestyle='--', linewidth=1.3)

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
file_path = os.path.join('C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Report\\Images', 'Mink_r_t.pdf')
plt.savefig(file_path)
plt.show()