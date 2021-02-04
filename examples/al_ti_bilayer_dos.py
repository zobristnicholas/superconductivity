# This file simulates the Al/Ti bilayer discussed in Zhao et al. 2018
# (doi:10.1088/1361-6668/aa94b7). In this paper they assume that the
# coherence length is an independent parameter, where in our code we
# assume that it's related to the resistivity from the equation in Martinis
# 2000 (doi:10.1016/S0168-9002(99)01320-0). We correct for this by
# specifying xi after the instantiation of the Superconductor object, but
# this is not considered supported behavior of the code.
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import e, hbar, k
from superconductivity.multilayer import Stack, Superconductor

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ti parameters
# thickness [m] 
d_ti = [25e-9, 45e-9, 80e-9, 105e-9, 130e-9]
# transition temperature [K]
tc_ti = 0.4
# Debye temperature [K]
td_ti = 420  # not actually used replaced by nc later
# diffusion constant [m^2 / s]
dc_ti = 2 * np.pi * 110e-9**2 * tc_ti * k / hbar  # using coherence length
# resistivity [Ohm m]
rho_ti = 1 / 3 * 1e-6

# Al parameters
# thickness [m]
d_al = 50e-9
# transition temperature [K]
tc_al = 1.2
# Debye temperature [K]
td_al = 433
# diffusion constant [m^2 / s]
dc_al = 2 * np.pi * 170e-9**2 * tc_al * k / hbar  # using coherence length
# resistivity [Ohm m]
rho_al = 1 / 180 * 1e-6

# Other parameters
gamma_b1 = 0.01 * np.sqrt(tc_al / tc_ti)  # dimensionless boundary resistance
t = 0.1  # temperature [K]

# Simulation
# Define the superconductors and put them into a bilayer.
al = Superconductor(d_al, rho_al, t, td_al, tc_al, dc_al)
ti = Superconductor(d_ti[0], rho_ti, t, td_ti, tc_ti, dc_ti)
stack = Stack([al, ti], gamma_b1)

# Do the simulation.
stack.update()

# Plot the density of states
stack.plot_dos()
plt.show(block=True)
