import numpy as np

def Aexprt(r):
    Q = 0
    mass = 0
    return 1-((2*mass)/r)+(Q**2)/(r**2)

def hexprt(r):
    Ccmc = 0
    Kcmc = -1
    return (-(Ccmc/r**2)-(Kcmc*r/3))/(Aexprt(r)*np.sqrt(Aexprt(r)+(-(Ccmc/r**2)-(Kcmc*r/3))**2))

def alpha_b(r):
    return np.sqrt(Aexprt(r)/(1-(Aexprt(r)*hexprt(r))**2))

def beta_b(r):
    return -(Aexprt(r)**2)*hexprt(r)/(1-(Aexprt(r)*hexprt(r))**2)
