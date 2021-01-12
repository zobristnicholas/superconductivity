import numpy as np
from scipy.constants import e
from scipy.special import digamma


class Metal:
    def __init__(self, d, rho, t, td, dc=None, *, n0=None):
        self.d = d  # thickness
        self.rho = rho  # resistivity
        self.td = td  # Debye temperature
        self.t = t  # temperature

        # The diffusion constant may be specified or derived from the
        # resistivity and the density of states. We save n0 or compute it
        # from the diffusion constant even though it won't be used for
        # calculations.
        if (n0 is None and dc is None) or (n0 is not None and dc is not None):
            raise ValueError("Supply only one of the 'n0' and 'dc' keyword "
                             "arguments.")
        if dc is None:
            self.dc = 1 / (self.rho * n0 * e ** 2)
            self.n0 = n0
        else:
            self.dc = dc
            self.n0 = 1 / (self.rho * self.dc * e ** 2)

        # Initialize the solution grid placeholders.
        self.z = None
        self.e = None
        self.order = None
        self.mtheta = None
        self.theta = None

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
        # Initialize the Matsubara energies.
        self.mtheta = np.zeros((self.nc + 1, self.z.size))

        # Initialize the order parameter.
        self.update_order()

        # Initialize the pair angle.
        self.theta = np.zeros((self.e.size, self.z.size))

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        self.order = (2 * np.pi * self.t * np.sum(np.sin(self.mtheta), axis=0)
                      / (digamma(self.nc + 1.5) - digamma(0.5)))
