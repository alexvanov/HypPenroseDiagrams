import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
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

alpha_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\pentrevol_ev_alpha_c4_single.yg"
alpha_data = read_data(alpha_path)
beta_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\pentrevol_ev_betar_c4_single.yg"
beta_data = read_data(beta_path)
chi_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\pentrevol_ev_chi_c4_single.yg"
chi_data = read_data(chi_path)
grr_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\pentrevol_ev_grr_c4_single.yg"
grr_data = read_data(grr_path)
gthth_path = "C:\\Users\\jacma\\OneDrive - Universidade de Lisboa\\Desktop\\PIC\\Penrose Diagrams\\Alex\\HypPenroseDiagrams-main\\data\\data\\pentrevol_ev_grr_c4_single.yg"
gthth_data = read_data(gthth_path)

#alpha
def alpha_p(r):
    alpha_time = 0.0000000000
    alpha_x_values, alpha_y_values = select_time_data(alpha_time, alpha_data)
    alpha_x_values_flat = alpha_x_values.flatten()
    alpha_y_values_flat = alpha_y_values.flatten()
    alpha_cs = interp1d(alpha_x_values_flat, alpha_y_values_flat, kind='cubic', fill_value="extrapolate")
    return alpha_cs(r)

#beta
def beta_p(r):
    beta_time = 0.0000000000
    beta_x_values, beta_y_values = select_time_data(beta_time, beta_data)
    beta_x_values_flat = beta_x_values.flatten()
    beta_y_values_flat = beta_y_values.flatten()
    beta_cs = interp1d(beta_x_values_flat, beta_y_values_flat, kind='cubic', fill_value="extrapolate")
    return beta_cs(r)

#chi
def chi_p(r):
    chi_time = 0.0000000000
    chi_x_values, chi_y_values = select_time_data(chi_time, chi_data)
    chi_x_values_flat = chi_x_values.flatten()
    chi_y_values_flat = chi_y_values.flatten()
    chi_cs = interp1d(chi_x_values_flat, chi_y_values_flat, kind='cubic', fill_value="extrapolate")
    return chi_cs(r)

#grr
def grr_p(r):
    grr_time = 0.0000000000
    grr_x_values, grr_y_values = select_time_data(grr_time, grr_data)
    grr_x_values_flat = grr_x_values.flatten()
    grr_y_values_flat = grr_y_values.flatten()
    grr_cs = interp1d(grr_x_values_flat, grr_y_values_flat, kind='cubic', fill_value="extrapolate")
    return grr_cs(r)

#gthth
def gthth_p(r):
    gthth_time = 0.0000000000
    gthth_x_values, gthth_y_values = select_time_data(gthth_time, gthth_data)
    gthth_x_values_flat = gthth_x_values.flatten()
    gthth_y_values_flat = 1 / np.sqrt(gthth_y_values.flatten())
    gthth_cs = interp1d(gthth_x_values_flat, gthth_y_values_flat, kind='cubic', fill_value="extrapolate")
    return gthth_cs(r)

'''
r = np.arange(0,1,0.0001)
plt.plot(r,alpha_p(r))
plt.plot(r,beta_p(r))
plt.plot(r,chi_p(r))
plt.plot(r,grr_p(r))
plt.plot(r,gthth_p(r))
plt.show()
'''