import numpy as np
from scipy.constants import k, hbar

from superconductivity.multilayer.pde import solve_diffusion_equation
from superconductivity.multilayer.superconductor import Superconductor


class Stack:
    """
    A collection of materials and boundary resistances which create the
    superconducting multilayer.
    """
    RTOL = 1e-8  # relative convergence tolerance
    MAX_ITERATIONS = 100  # maximum number of iterations to converge
    SPEEDUP = 10  # every <SPEEDUP> iteration is a Steffensen iteration
    THRESHOLD = 1e-3  # DOS threshold for determining the gap energy
    N_GRID = 5  # position grid points per layer

    def __init__(self, layers, boundaries):
        if len(boundaries) != len(layers) - 1:
            raise ValueError("There must be a boundary term for each layer "
                             "interface (one less than the number of layers).")
        self._z = None
        self._e = None
        self.layers = list(layers)  # from bottom to top
        self.boundaries = list(boundaries)

        # Update the energy scale with which to normalize the calculations.
        self._update_scale()

        # Set up the default grid for the calculations.
        self._update_grid()

    @property
    def d(self):
        """The total thickness of the stack."""
        return sum([layer.d for layer in self.layers])

    @property
    def z(self):
        """The position solution grid for the simulation."""
        return self._z

    @z.setter
    def z(self, positions):
        self._z = positions
        d = 0
        for layer in self.layers:
            layer.z = positions[(positions >= d) & (positions <= layer.d + d)]
            d += layer.d

    @property
    def e(self):
        """The energy solution grid for the simulation."""
        return self._e

    @e.setter
    def e(self, energies):
        self._e = energies
        for layer in self.layers:
            layer.e = energies

    def add_top(self, layer, boundary):
        """
        Add a layer to the top of the stack. The boundary term should be
        referenced to the resistivity and coherence length of the layer
        below. Using this method will reset the solution grid to the
        default settings.
        """
        self.layers.append(layer)
        self.boundaries.append(boundary)
        self._update_scale()
        self._update_grid()

    def add_bottom(self, layer, boundary):
        """
        Add a layer to the bottom of the stack. The boundary term should
        be referenced to the resistivity and coherence length of the
        layer being added. Using this method will reset the solution
        grid to the default settings.
        """
        self.layers.insert(0, layer)
        self.boundaries.insert(0, boundary)
        self._update_scale()
        self._update_grid()

    def update_order(self):
        """
        Update the order parameter for the entire stack by solving the
        Usadel diffusion equation and self-consistency equation.
        """
        # Initialize all of the parameters to their bulk values
        temperatures = []
        for layer in self.layers:
            layer.initialize_bulk()
            temperatures.append(layer.t)

        # Enforce a constant temperature
        if not all([temperatures[0] == t for t in temperatures]):
            raise ValueError("All materials must have the same temperature")

        # Iterate between solving for the order and pair angle until the
        # order converges or we run out of iterations.
        r = np.inf
        i = 0
        nmax = max([m.nc for m in self.layers])
        wn = (2 * np.arange(0, nmax + 1) + 1) * np.pi * k * self.layers[0].t
        while r > self.RTOL and i < self.MAX_ITERATIONS:
            # Update the iteration counter.
            i += 1

            # Loop through the Matsubara frequencies.
            for ii in range(nmax + 1):
                mtheta = self.solve_diffusion_equation(1j * wn[ii])
                for iii, layer in enumerate(self.layers):
                    layer.mtheta[ii, :] = mtheta[iii]

        # Update the order with the new pair angle.
        for layer in self.layers:
            layer.update_order()

    def solve_diffusion_equation(self, energy):
        # Normalize all of the parameters and get them ready to go into the
        # Fortran PDE solving routine.
        energy = energy / self.scale
        order = np.array([m.order for m in self.layers])
        d = np.array([m.d for m in self.layers])
        x_scale = np.array([(2 * self.scale * m.d**2 / (hbar * m.dc))
                            for m in self.layers])
        return solve_diffusion_equation(energy, self.z, order, d, x_scale)

    def _update_scale(self):
        tcs = [m.tc for m in self.layers if isinstance(l, Superconductor)]
        self.scale = np.pi * k * max(tcs) if tcs else 1

    def _update_grid(self):
        start, stop = [], []
        for layer in layers:
            start.append(sum(stop))
            stop.append(layer.d + sum(stop))
        positions = np.concatenate([np.linspace(start[i], stop[i], 5)
                                    for i in range(len(layers))])
        self.z = positions
        energies = np.linspace(0, 2 * self.scale, 1000)
        self.energies = energies
