import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
    # print(f"read {data.nbytes / 1024:.2f} KByte of data from {filename};")
    
    return times, data

# File locations
base_path = "C:/Users/jacma/OneDrive - Universidade de Lisboa/Desktop/PIC/Penrose Diagrams/Alex/HypPenroseDiagrams-main/data/data"

files_m4new = [
    os.path.join(base_path, "pentrevolm4new_diag_R_c4_single.yg"),
    os.path.join(base_path, "pentrevolm4new_diag_T_c4_single.yg")
]

files_m5new = [
    os.path.join(base_path, "pentrevolm5new_diag_R_c4_single.yg"),
    os.path.join(base_path, "pentrevolm5new_diag_T_c4_single.yg")
]

files_m07pert = [
    os.path.join(base_path, "pentrevolm07_diag_R_c4_single.yg"),
    os.path.join(base_path, "pentrevolm07_diag_T_c4_single.yg")
]

files_0pert = [
    os.path.join(base_path, "pentrevol0_diag_R_c4_single.yg"),
    os.path.join(base_path, "pentrevol0_diag_T_c4_single.yg")
]

# Function to process data
def process_data(files):
    vard = [ReadPygraph(file) for file in files]
    
    data = []
    for ct in range(len(vard[0][0])):
        inner_data = []
        for st in range(len(vard[0][1][ct])):
            inner_data.append([
                vard[0][1][ct][st][1] / 2,
                vard[1][1][ct][st][1] / 2
            ])
        data.append([vard[0][0][ct], inner_data])
    return data

# Function to process perturbed data
def process_perturbed_data(files):
    vard = [ReadPygraph(file) for file in files]
    
    data = []
    for ct in range(len(vard[0][0])):
        inner_data = []
        for st in range(len(vard[0][1][ct])):
            inner_data.append([
                vard[0][1][ct][st][1],
                vard[1][1][ct][st][1]
            ])
        data.append([vard[0][0][ct], inner_data])
    return data

# Process data for each case
diagdatam4new = process_data(files_m4new)
diagdatam5new = process_data(files_m5new)
diagdatam07pert = process_perturbed_data(files_m07pert)
diagdatam0pert = process_perturbed_data(files_0pert)

def plot_data(data, title):
    # Extract every 5th element
    data = data[::5]
    
    for entry in data:
        x = np.array([point[0] for point in entry[1]])
        y = np.array([point[1] for point in entry[1]])
        plt.plot(x, y)
    
    plt.xlim(0, np.pi/2)
    plt.ylim(-np.pi/2, np.pi/2)
    plt.gca().set_aspect(1)
    plt.title(title)
    plt.show()

# Plot
plot_data(diagdatam4new, 'diagdatam4new')
plot_data(diagdatam5new, 'diagdatam5new')
plot_data(diagdatam07pert, 'diagdatam07pert')
plot_data(diagdatam0pert, 'diagdata0pert')


########################################################################

# Comparison between initially CMC and perturbed slices

def plot_comparison(data1, data2):
    fig, ax = plt.subplots()
    first_iteration1= True
    first_iteration2= True

    data1 = data1[::10]
    for entry in data1:
        x = np.array([point[0] for point in entry[1]])
        y = np.array([point[1] for point in entry[1]])
        if first_iteration1:
            ax.plot(x, y, color='blue', label='diagdatam5new')
            first_iteration1 = False
        else: ax.plot(x, y, color='blue')

    data2 = data2[::10]
    for entry in data2:
        x = np.array([point[0] for point in entry[1]])
        y = np.array([point[1] for point in entry[1]])
        if first_iteration2:
            ax.plot(x, y, color='black', label='diagdatam07pert')
            first_iteration2 = False
        else: ax.plot(x, y, color='black')

    ax.set_xlim(0, np.pi/2)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_aspect(1)
    plt.legend()
    plt.show()

plot_comparison(diagdatam5new, diagdatam07pert)

##############################################################################


# Initial data start from a perturbed Minkowski slice

def covercol(ax, colpoint):
    ax.plot([0, np.pi/2], [-np.pi/2, 0], 'k-', linewidth=2)
    ax.plot([(np.pi/2 - colpoint)/2, np.pi/2], [colpoint/2 + np.pi/4, 0], 'k-', linewidth=2)
    ax.plot([0, (np.pi/2 - colpoint)/2], [colpoint, colpoint + (np.pi/2 - colpoint)/2], 'k-', linewidth=2)
    ax.plot([0, (np.pi/2 - colpoint)/2], [colpoint/2 + np.pi/4, colpoint/2 + np.pi/4], 'k-', linewidth=2)
    ax.plot([0, 0], [-np.pi/2, colpoint/2 + np.pi/4], 'k-', linewidth=2)

    ax.set_xlim(-0.5, np.pi/2 + 0.07)
    ax.set_ylim(-np.pi/2-0.03, colpoint/2 + np.pi/4+0.03)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Add labels
    ax.text((np.pi/2 - colpoint)/2 + 0.05, colpoint/2 + np.pi/4 + 0.05, r"$i^+$", fontsize=18)
    ax.text(0 + 0.05, -np.pi/2 - 0.07, r"$i^-$", fontsize=18)
    ax.text(np.pi/2 + 0.05, 0, r"$i^0$", fontsize=18)
    ax.text(((np.pi/2 - colpoint)/2 + np.pi/2)/2 + 0.05, (colpoint/2 + np.pi/4)/2 + 0.05, r"$ \scri ^{+}$", fontsize=18)
    ax.text(np.pi/4 + 0.1, -np.pi/4 - 0.1, r"$ \scri ^{-}$", fontsize=18)
    ax.text((np.pi/2 - colpoint)/4-0.1, colpoint/2 + np.pi/4 + 0.07, r"$\tilde{r}=0$", fontsize=18)


# Plot
fig, ax = plt.subplots()
covercol(ax, 0.55)
plot_data(diagdatam0pert, 'diagdatam0pert')
plt.close(fig)

fig, ax = plt.subplots()
covercol(ax, -0.08)
plot_data(diagdatam07pert, 'diagdatam07pert')
plt.close(fig)

fig, ax = plt.subplots()
covercol(ax, 0.7)
plot_data(diagdatam4new, 'diagdatam4new')
plt.close(fig)

fig, ax = plt.subplots()
covercol(ax, -0.15)
plot_data(diagdatam5new, 'diagdatam5new')
plt.close(fig)

# Creat the animation

def update_plot(ti):
    ax.clear()
    covercol(ax, -0.15)
    colpoint = -0.15
    entry = diagdatam5new[ti]
    x = np.array([point[0] for point in entry[1]])
    y = np.array([point[1] for point in entry[1]])
    ax.plot(x, y, color='blue', linewidth=2)
    ax.set_xlim(-0.05, np.pi/2 + 0.03)
    ax.set_ylim(-np.pi/2-0.03, colpoint/2 + np.pi/4+0.03)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

fig, ax = plt.subplots()
anim = FuncAnimation(fig, update_plot, frames=len(diagdatam5new), repeat=False)
gif_path = "C:/Users/jacma/OneDrive - Universidade de Lisboa/Desktop/PIC/Gifs/colm5.gif"
anim.save(gif_path, writer=PillowWriter(fps=30))
plt.show()