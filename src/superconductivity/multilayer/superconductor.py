import logging
import numpy as np
from scipy.special import digamma
from scipy.optimize import brentq
from scipy.constants import k, hbar

from superconductivity.utils import BCS
from superconductivity.multilayer.metal import Metal
from superconductivity.gap_functions import delta_bcs
from superconductivity.density_of_states import usadel_pair_angle


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
        alpha: float (optional)
            The disorder dependent pair breaking parameter normalized to
            the zero temperature gap energy with no pair breaking. See Coumou
            et al. (doi:10.1103/PhysRevB.88.180505) for more details.
    """
    def __init__(self, d, rho, t, td, tc, dc=None, *, n0=None, alpha=0.):
        # Set the debye temperature before initializing the base class since
        # the base class sets t. t has been overloaded with a property that
        # requires td.
        self.td = td  # Debye temperature

        # Initialize the base class.
        super().__init__(d, rho, t, dc=dc, n0=n0)

        self.tc = tc  # transition temperature
        self.alpha = alpha * BCS * k * self.tc  # pair breaking parameter

        # The zero temperature order parameter is derived from the transition
        # temperature assuming a BCS superconductor.
        self.delta0 = BCS * k * self.tc

        # The coherence length in the dirty limit is derived from the
        # diffusion constant and the zero temperature gap energy.
        self.xi = np.sqrt(BCS * hbar * self.dc / (2 * np.pi * self.delta0))

        # Make more appropriate energy and position grids.
        self.z = np.linspace(0.0, self.d, max(10, int(10 * self.d / self.xi)))
        self.e *= (BCS * self.tc / self.t)

        # Define computation parameters for bulk computation when alpha != 0.
        self.rtol = 1e-4  # relative convergence tolerance
        self.max_iterations = 100  # maximum number of iterations to converge
        self.threshold = 1e-3  # DOS threshold for determining the gap energy

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
        log.debug("Initializing bulk parameters")
        # Initialize the gap.
        g = delta_bcs(self.t, self.tc, interp=True, approx=True)
        self.gap = np.full(self.z.shape, g)

        # Initialize the pair angle at the Matsubara energies.
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

        # Update estimates if alpha is nonzero
        if self.alpha != 0:
            # Initialize the while loop parameters.
            r = np.inf
            i = 0
            last_order = self.order[0] / self.delta0

            # Compute the order parameter by iterating until convergence.
            log.debug("Computing the order parameter.")
            while r > self.rtol and i < self.max_iterations:
                # Update the iteration counter.
                i += 1

                # Solve for the pair angle first
                mtheta = usadel_pair_angle(1j * wn, self.order[0], self.alpha)
                self.mtheta[:] = mtheta

                # Update the order parameter using the new pair angle.
                self.update_order()

                # Save iteration history and evaluate convergence criteria.
                new_order = self.order[0] / self.delta0
                r = np.max(np.abs(new_order - last_order) / (
                        np.abs(new_order) + 1))
                last_order = self.order[0] / self.delta0
                log.debug("Iteration: {:d} :: R: {:g}".format(i, r))

            # Compute the pair angle with the new order parameter.
            log.debug("Computing the pair angle.")
            theta = usadel_pair_angle(self.e, self.order[0], self.alpha)
            self.theta[:] = theta[:, np.newaxis]

            # Compute the energy gap.
            log.debug("Computing the gap energy.")

            def find_gap(e):
                en = e * self.order[0]
                th = usadel_pair_angle(en, self.order[0], self.alpha)
                return np.cos(th).real - self.threshold

            dos = np.cos(self.theta[:, 0]).real
            try:
                max_e = self.e[dos > self.threshold].min() / self.order[0]
                min_e = self.e[dos < self.threshold].max() / self.order[0]
                self.gap[:] = brentq(find_gap, min_e, max_e) * self.order[0]
            except ValueError:  # the bounds didn't give opposite signs
                max_e = np.max(self.e) / self.order[0]
                try:
                    self.gap[:] = brentq(find_gap, 0, max_e) * self.order[0]
                except ValueError:
                    self.gap[:] = 0.
        log.debug("Bulk parameters initialized.")

    def update_order(self):
        """
        Update the order parameter from the pair angle at the Matsubara
        frequencies.
        """
        tr = self.t / self.tc
        self.order = (2 * np.pi * k * self.t
                      * np.sum(np.sin(self.mtheta), axis=0)
                      / (np.log(tr) + digamma(self.nc + 1.5) - digamma(0.5)))
