import numpy as np
from scipy.special import digamma
from scipy.constants import k, hbar

from superconductivity.utils import BCS
from superconductivity.multilayer.metal import Metal
from superconductivity.gap_functions import delta_bcs


class Superconductor(Metal):
    def __init__(self, d, rho, t, td, tc, dc=None, *, n0=None):
        super().__init__(d, rho, t, td, dc=dc, n0=n0)
        self.tc = tc

        # The zero temperature gap energy is derived from the transition
        # temperature assuming a BCS superconductor.
        self.delta0 = BCS * k * self.tc

        # The coherence length in the dirty limit is derived from the
        # diffusion constant and the zero temperature gap energy.
        self.xi = np.sqrt(BCS * hbar * self.dc / (2 * np.pi * self.delta0))

        self.gap = None

        # Rescale the energy to a more valid range.
        self.e *= 2 * np.pi * k * self.tc

    def initialize_bulk(self):
        """
        Initialize calculation parameters to their bulk values at the
        grid locations.
        """
        # Initialize the gap.
        g = delta_bcs(self.t, self.tc, interp=True, approx=True)
        self.gap = np.broadcast_to(g, self.z.shape)

        # Initialize the Matsubara energies.
        self.wn = (2 * np.arange(0, self.nc + 1) + 1) * np.pi * k * self.t
        self.wn = self.wn[:, np.newaxis]
        self.mtheta = np.arcsin(self.gap / np.sqrt(self.gap**2 + self.wn**2))

        # Initialize the order parameter.
        self.update_order()

        # Initialize the pair angle.
        self.theta = np.empty((self.E_GRID, self.Z_GRID), dtype=complex)
        nonzero = (self.e != 0)
        self.theta[nonzero, :] = np.arctan(1j * self.gap
                                           / self.e[nonzero, np.newaxis])
        self.theta[~nonzero, :] = np.pi / 2

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        tr = self.t / self.tc
        self.order = (2 * np.pi * k * self.t
                      * np.sum(np.sin(self.mtheta), axis=0)
                      / (np.log(tr) + digamma(self.nc + 1.5) - digamma(0.5)))
