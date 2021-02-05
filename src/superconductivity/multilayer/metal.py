import numpy as np
from scipy.constants import e
from scipy.special import digamma


class Metal:
    """
    A class holding the material properties of a metal.

    Args:
        d: float
            The thickness of the material [m].
        rho: float
            The normal state resistivity of the material [Ohm m].
        t: float
            The temperature of the material in [K].
        td: float
            The Debye temperature of the material [K].
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
    Z_GRID = 5  # position grid points
    E_GRID = 1000  # energy grid points

    def __init__(self, d, rho, t, td, dc=None, *, n0=None):
        self.d = d  # thickness
        self.rho = rho  # resistivity
        self.td = td  # Debye temperature
        self.t = t  # temperature

        # Initialize the solution grid placeholders.
        self.z = np.linspace(0.0, self.d, self.E_GRID)
        # There is no natural energy scale for a simple metal.
        self.e = np.linspace(0.0, 1.0, self.E_GRID)
        self.wn = None
        self.order = None
        self.mtheta = None
        self.theta = None

        # The diffusion constant may be specified or derived from the
        # resistivity and the density of states. We save n0 or compute it
        # from the diffusion constant even though it won't be used for
        # calculations.
        if (n0 is None and dc is None) or (n0 is not None and dc is not None):
            raise ValueError("Supply only one of the 'n0' and 'dc' keyword "
                             "arguments.")
        if dc is None:
            self.dc = 1 / (self.rho * n0 * e**2)
            self.n0 = n0
        else:
            self.dc = dc
            self.n0 = 1 / (self.rho * self.dc * e**2)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, temperature):
        self._t = temperature
        # The Matsubara cutoff integer is derived from the temperature and
        # the Debye temperature.
        self.nc = int(np.floor(self.td / (2 * np.pi * self.t) - 0.5))

    def initialize_bulk(self):
        """
        Initialize calculation parameters to their bulk values at the
        grid locations.
        """
        # Initialize the Matsubara energies and pair angles.
        self.wn = (2 * np.arange(0, self.nc + 1) + 1) * np.pi * k * self.t
        self.mtheta = np.zeros((self.nc + 1, self.Z_GRID))

        # Initialize the order parameter.
        self.update_order()

        # Initialize the pair angle.
        self.theta = np.zeros((self.E_GRID, self.Z_GRID))

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        self.order = (2 * np.pi * k * self.t
                      * np.sum(np.sin(self.mtheta), axis=0)
                      / (digamma(self.nc + 1.5) - digamma(0.5)))