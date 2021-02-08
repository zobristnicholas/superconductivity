# This file simulates the Al/Ti/Al trilayer discussed in Zhao et al. 2018
# (doi:10.1088/1361-6668/aa94b7) and reproduces figure 3.
import logging
import numpy as np
from scipy.constants import hbar, k, e
from matplotlib import pyplot as plt
from superconductivity.gap_functions import delta_bcs
from superconductivity.multilayer import Stack, Superconductor

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ti parameters
# thickness [m]
d_ti = 100e-9
# transition temperature [K]
tc_ti = 0.55
# Debye temperature [K]
td_ti = 426  # not actually used since we fix nc later
# coherence length [m]
xi_ti = 110e-9
# diffusion constant [m^2 / s]
dc_ti = 1.5
# resistivity [Ohm m]
rho_ti = 1 / 5.88 * 1e-6

# Al parameters
# thickness [m]
d_al = 100e-9
# transition temperature [K]
tc_al = 1.2
# Debye temperature [K]
td_al = 423  # not actually used since we fix nc later
# coherence length [m]
xi_al = 170e-9
# diffusion constant [m^2 / s]
dc_al = 35
# resistivity [Ohm m]
rho_al = 1 / 132 * 1e-6

# Other parameters
rb = 0.0 * rho_al * xi_al  # boundary resistance [Ohm m^2]
min_t = min(tc_al, tc_ti) / 10
max_t = max(tc_al, tc_ti)
t = np.linspace(min_t, max_t, 100)  # temperature [K]

# Simulation
# Define the superconductors
al = Superconductor(d_al, rho_al, t[0], td_al, tc_al, dc_al)
ti = Superconductor(d_ti, rho_ti, t[0], td_ti, tc_ti, dc_ti)

# We need a larger grid for the BVP to be solvable in this simulation
Superconductor.Z_GRID = 10

# Add the superconductors to a stack
stack = Stack([al, ti], rb)

# Loop over the temperatures
figure, axes = plt.subplots()
gap = np.zeros(len(t))
for ii, tii in enumerate(t):
    # Set the new temperature
    stack.t = tii

    # Do the simulation
    stack.update_order()
    stack.update_gap()

    # Save the gap energy (same for all layers and positions)
    gap[ii] = stack.layers[0].gap.mean()

# Plot the results.
plt.plot(t, gap / e * 1e3, label="bilayer")
plt.plot(t, delta_bcs(t, tc_al) / e * 1e3, label='bulk Al')
plt.plot(t, delta_bcs(t, tc_ti) / e * 1e3, label='bulk Ti')
plt.xlabel("temperature [K]")
plt.ylabel("gap energy [meV]")
plt.legend()
plt.show(block=True)
