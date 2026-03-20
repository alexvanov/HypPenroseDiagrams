import sympy as sp
import numpy as np

# Define the symbolic variables and constants
r, Ccmc = sp.symbols('r, Ccmc')
Kcmc = -1 
mass = 1
Q = 0
aconf = 1
aconf_prime = sp.diff(aconf, r)

# Define the equation
equation = (9*r**4*(aconf-r*aconf_prime)**2) / (9*Ccmc**2*aconf**6+9*Q**2*aconf**4*r**2+ \
              6*(Ccmc*Kcmc-3*mass)*aconf**3*r**3+ \
              9*aconf**2*r**4+Kcmc**2*r**6)

eq = r**4/equation
discriminant = sp.discriminant(eq, r)

# Solve the condition where discriminant == 0 for Ccmc
ccmc_solutions = sp.solve(discriminant, Ccmc)
valid_ccmc_solutions = [sol for sol in ccmc_solutions if sp.im(sol) == 0 and sol != 0]
evaluated_ccmc = [Ccmc.evalf() for Ccmc in valid_ccmc_solutions]

#print("Valid Ccmc solutions:", evaluated_ccmc)

def Ccmc_crit(x):
    return float(evaluated_ccmc[-1])