# This file simulates the Al/Ti/Al trilayer discussed in Zhao et al. 2018
# (doi:10.1088/1361-6668/aa94b7).
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import e, hbar, k
from superconductivity.multilayer import Stack, Superconductor

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ti parameters
# thickness [m]
d_ti = [25e-9, 50e-9, 100e-9, 150e-9, 200e-9]
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
# coherence length [m]
xi_al = 170e-9
# diffusion constant [m^2 / s]
dc_al = 2 * np.pi * xi_al**2 * tc_al * k / hbar  # using coherence length
# resistivity [Ohm m]
rho_al = 1 / 180 * 1e-6

# Other parameters
rb = 0.01 * rho_al * xi_al  # dimensionless boundary resistance
t = 0.1  # temperature [K]

# Simulation
# Define the superconductors
al = Superconductor(d_al, rho_al, t, td_al, tc_al, dc_al)
ti = Superconductor(d_ti[-1], rho_ti, t, td_ti, tc_ti, dc_ti)

# Set some simulation parameters to match those used in the paper
Stack.RTOL = 1e-3
Stack.SPEEDUP = 0
Superconductor.Z_GRID = 10
al.nc = 125
ti.nc = 125

# Add the superconductors to the trilayer
stack = Stack([al, ti, al], [rb, rb])

# Do the simulation
stack.update()
# stack.update_order()
# Plot
stack.plot(location='mixed')
plt.show(block=True)
