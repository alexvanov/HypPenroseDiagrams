# Data with pretag "imitateoldoslong_"

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def read_data(file_path):
    data = {'Time': [], 'rcomp': [], 'rphy': []}
    current_time = None
    with open(file_path, 'r') as file:
        x_values = []
        y_values = []
        for line in file:
            line = line.strip()
            if line.startswith("#Time"):
                if current_time is not None:
                    data['Time'].append(current_time)
                    data['rcomp'].append(np.array(x_values))
                    data['rphy'].append(np.array(y_values))
                current_time = float(line.split("=")[1])
                x_values = []
                y_values = []
            elif line:  # Check if line is not empty
                x_value, y_value = map(float, line.split())
                x_values.append(x_value)
                y_values.append(y_value)
        # Append the last set of values
        if current_time is not None:
            data['Time'].append(current_time)
            data['rcomp'].append(np.array(x_values))
            data['rphy'].append(np.array(y_values))
    return data


def select_time_data(time, data):    
    time_index = data['Time'].index(time)
    x_values = np.array([data['rcomp'][time_index]])
    y_values = np.array([data['rphy'][time_index]])
    return x_values, y_values

alpha_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\imitateoldoslong_ev_alpha_c4_single.yg"
alpha_data = read_data(alpha_path)
beta_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\imitateoldoslong_ev_betar_c4_single.yg"
beta_data = read_data(beta_path)
chi_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\imitateoldoslong_ev_chi_c4_single.yg"
chi_data = read_data(chi_path)
grr_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\imitateoldoslong_ev_grr_c4_single.yg"
grr_data = read_data(grr_path)
gthth_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\imitateoldoslong_ev_grr_c4_single.yg"
gthth_data = read_data(gthth_path)

#alpha
def alpha_num(r):
    #alpha_time = 0.0000000000
    alpha_time = 50.0000000001
    alpha_x_values, alpha_y_values = select_time_data(alpha_time, alpha_data)
    alpha_x_values_flat = alpha_x_values.flatten()
    alpha_y_values_flat = alpha_y_values.flatten()
    alpha_cs = CubicSpline(alpha_x_values_flat, alpha_y_values_flat)
    return alpha_cs(r)

#beta
def beta_num(r):
    #beta_time = 0.0000000000
    beta_time = 50.0000000001
    beta_x_values, beta_y_values = select_time_data(beta_time, beta_data)
    beta_x_values_flat = beta_x_values.flatten()
    beta_y_values_flat = beta_y_values.flatten()
    beta_cs = CubicSpline(beta_x_values_flat, beta_y_values_flat)
    return beta_cs(r)

#chi
def chi_num(r):
    #chi_time = 0.0000000000
    chi_time = 50.0000000001
    chi_x_values, chi_y_values = select_time_data(chi_time, chi_data)
    chi_x_values_flat = chi_x_values.flatten()
    chi_y_values_flat = chi_y_values.flatten()
    chi_cs = CubicSpline(chi_x_values_flat, chi_y_values_flat)
    return chi_cs(r)

#grr
def grr_num(r):
    #grr_time = 0.0000000000
    grr_time = 50.0000000001
    grr_x_values, grr_y_values = select_time_data(grr_time, grr_data)
    grr_x_values_flat = grr_x_values.flatten()
    grr_y_values_flat = grr_y_values.flatten()
    grr_cs = CubicSpline(grr_x_values_flat, grr_y_values_flat)
    return grr_cs(r)

#gthth
def gthth_num(r):
    #gthth_time = 0.0000000000
    gthth_time = 50.0000000001
    gthth_x_values, gthth_y_values = select_time_data(gthth_time, gthth_data)
    gthth_x_values_flat = gthth_x_values.flatten()  # transform x-values
    gthth_y_values_flat = 1 / np.sqrt(gthth_y_values.flatten())  # transform y-values
    gthth_cs = CubicSpline(gthth_x_values_flat, gthth_y_values_flat)  # create cubic spline
    return gthth_cs(r)

def omega(r):
    rscri = 1
    Kcmc = -1
    return (-r**2+rscri**2)/(2*(-3/Kcmc)*rscri)


def aconf_num(r):
    return np.sqrt(chi_num(r)/gthth_num(r))*omega(r)

'''
r = np.arange(0, 1, 0.001)
plt.plot(r,alpha_num(r))
plt.plot(r,beta_num(r))
plt.plot(r,chi_num(r))
plt.plot(r,grr_num(r))
plt.plot(r,gthth_num(r))
plt.show()
'''

# Calculate c

def num_derivative(f, x, h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)

def expression(r):
    numerator = grr_num(r) / chi_num(r)
    denominator = (1/alpha_num(r)**2)*(omega(r)**2/aconf_num(r)**2*(aconf_num(r)-r*num_derivative(aconf_num,r)))**2
    return numerator/denominator

const = expression(0.5)
c = np.sqrt(const)
print(f"Value of c taken to be c = {c} -- It must be 1 if CMC data is being used.")


