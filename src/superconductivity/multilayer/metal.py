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
    def __init__(self, d, rho, t, dc=None, *, n0=None):
        self.d = d  # thickness
        self.rho = rho  # resistivity
        self.t = t  # temperature

        # Define computation parameters.
        self.z_grid = 5  # position grid points
        self.e_grid = 1000  # energy grid points

        # Initialize the solution grid placeholders.
        self.z = np.linspace(0.0, self.d, self.e_grid)
        # There is no natural energy scale for a simple metal.
        self.e = np.linspace(0.0, 1.0, self.e_grid)
        self.order = None
        self.mtheta = None
        self.theta = None
        self.gap = None

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

    def initialize_bulk(self):
        """
        Initialize calculation parameters to their bulk values at the
        grid locations.
        """
        # Initialize the Matsubara energies and pair angles.
        self.mtheta = np.zeros((1, self.z_grid))

        # Initialize the order parameter.
        self.order = np.zeros(self.z_grid)

        # Initialize the gap energy.
        self.gap = self.order

        # Initialize the pair angle.
        self.theta = np.zeros((self.e_grid, self.z_grid))

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        # The superconducting interaction potential is  proportional to the
        # order parameter. In a normal metal, the interaction potential is
        # zero, so the order parameter is always zero. We don't need to
        # update it here unless it hasn't been initialized.
        if self.order is None:
            self.order = np.zeros(self.z_grid)
