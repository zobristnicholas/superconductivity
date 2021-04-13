import numpy as np
from scipy.special import digamma
from scipy.constants import k, hbar

from superconductivity.utils import BCS
from superconductivity.multilayer.metal import Metal
from superconductivity.gap_functions import delta_bcs


class Superconductor(Metal):
    """
    A class holding the material properties of a superconductor.

    Args:
        d: float
            The thickness of the material [m].
        rho: float
            The normal state resistivity of the material [Ohm m].
        t: float
            The temperature of the material in [K].
        td: float
            The Debye temperature of the material [K].
        tc: float
            The superconducting transition temperature of the material
            [K].
        dc: float (optional)
            The diffusion constant of the material [m^2 / s]. If
            not provided, 'n0' must be provided.
        n0: float (optional)
            The single spin density of states [1 / (J m^3)]. If provided
            it is used to calculate 'dc' from the formula:
            1 / (rho * n0 * e**2). One and only one of dc and n0 should
            be provided.
            doi:10.1016/S0168-9002(99)01320-0
    """
    def __init__(self, d, rho, t, td, tc, dc=None, *, n0=None):
        # Set the debye temperature before initializing the base class since
        # the base class sets t. t has been overloaded with a property that
        # requires td.
        self.td = td  # Debye temperature

        # Initialize the base class.
        super().__init__(d, rho, t, dc=dc, n0=n0)

        self.tc = tc  # transition temperature

        # The zero temperature gap energy is derived from the transition
        # temperature assuming a BCS superconductor.
        self.delta0 = BCS * k * self.tc

        # The coherence length in the dirty limit is derived from the
        # diffusion constant and the zero temperature gap energy.
        self.xi = np.sqrt(BCS * hbar * self.dc / (2 * np.pi * self.delta0))

        # Make more appropriate energy and position grids.
        self.z = np.linspace(0.0, self.d, max(10, int(10 * self.d / self.xi)))
        self.e *= (BCS * self.tc / self.t / 2)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, temperature):
        self._t = temperature
        # The Matsubara cutoff integer is derived from the temperature and
        # the Debye temperature.
        self.nc = int(np.floor(self.td / (2 * np.pi * self._t) - 0.5))

    def initialize_bulk(self):
        """
        Initialize calculation parameters to their bulk values at the
        grid locations.
        """
        # Initialize the gap.
        g = delta_bcs(self.t, self.tc, interp=True, approx=True)
        self.gap = np.full(self.z.shape, g)

        # Initialize the Matsubara energies.
        wn = (2 * np.arange(0, self.nc + 1) + 1) * np.pi * k * self.t
        wn = wn[:, np.newaxis]
        self.mtheta = np.arcsin(self.gap / np.sqrt(self.gap**2 + wn**2))

        # Initialize the order parameter.
        self.update_order()

        # Initialize the pair angle.
        self.theta = np.empty((self.e.size, self.z.size), dtype=complex)
        zero = (self.e == 0)
        self.theta[~zero, :] = np.arctan(1j * self.gap
                                         / self.e[~zero, np.newaxis])
        self.theta[zero, :] = np.pi / 2

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        tr = self.t / self.tc
        self.order = (2 * np.pi * k * self.t
                      * np.sum(np.sin(self.mtheta), axis=0)
                      / (np.log(tr) + digamma(self.nc + 1.5) - digamma(0.5)))
