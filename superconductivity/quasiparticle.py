import numpy as np
import scipy.integrate as it

from superconductivity.utils import BCS
from superconductivity.fermi_functions import fermi
from superconductivity.density_of_states import dos_bcs, dos_dynes
from superconductivity.gap_functions import delta_bcs, reduced_delta_bcs, delta_dynes, reduced_delta_dynes, reduced_tc_dynes


def _integrand_bcs(x, t, dr, bcs):
    return dos_bcs(x, dr) * fermi(x, t / bcs)


def reduced_density_bcs(t, low_temp=False, bcs=BCS, **delta_kwargs):
    """
    The BCS quasiparticle density in units of 4 N0 ∆0 calculated either
    numerically or with the low temperature formula.
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    low_temp: boolean (optional)
        Determines whether to use the low temperature approximation. The
        default is False.
    bcs: boolean (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    delta_kwargs: optional keyword arguments
        Optional arguments to send to
        superconductivity.gap_functions.reduced_delta_bcs()
    Returns
    -------
    nqpr : float
        The reduced quasiparticle density.
    """
    nqpr = np.empty(t.shape)
    dr = reduced_delta_bcs(t, **delta_kwargs)
    zero = (t == 0)
    nqpr[zero] = 0
    if low_temp:
        dr_zero = (dr == 0)
        zero = zero | dr_zero
        nqpr[dr_zero] = np.inf
        nqpr[~zero] = np.sqrt(np.pi * t[~zero] / (2 * bcs * dr[~zero])) * np.exp(-bcs * dr[~zero] / t[~zero])
    else:
        nqpr_zero = nqpr[~zero]
        for ii, tii in enumerate(t[~zero]):
            nqpr_zero[ii] = it.quad(_integrand_bcs, dr[~zero][ii], np.inf, args=(tii, dr[~zero][ii], bcs))[0]
        nqpr[~zero] = nqpr_zero
    return nqpr


def density_bcs(temp, tc, n0, low_temp=False, bcs=BCS, **delta_kwargs):
    """
    The BCS quasiparticle density calculated either numerically or with the low
    temperature formula.
    Parameters
    ----------
    temp : float or numpy.ndarray
        The temperature in units of Kelvin.
    tc: float
        The transition temperature in Kelvin.
    n0: float
        The single spin density of states in units of states / Joule / volume.
    low_temp: boolean (optional)
        Determines whether to use the low temperature approximation. The
        default is False.
    bcs: boolean (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    delta_kwargs: optional keyword arguments
        Optional arguments to send to
        superconductivity.gap_functions.reduced_delta_bcs()
    Returns
    -------
    nqp : float
        The quasiparticle density in units of number / volume (see n0).
    """
    nqpr = reduced_density_bcs(temp / tc, low_temp=low_temp, bcs=bcs, **delta_kwargs)
    delta0 = delta_bcs(0, tc, bcs=bcs, **delta_kwargs)
    nqp = 4 * n0 * delta0 * nqpr
    return nqp


def _integrand_dynes(x, t, dr, bcs, g, tc):
    return dos_dynes(x, dr, g) * fermi(x, t / bcs * tc * (g + np.sqrt(g**2 + 1)))


def reduced_density_dynes(t, g, bcs=BCS, **delta_kwargs):
    """
    The Dynes quasiparticle density in units of 4 N0 ∆0 calculated numerically.
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    g: float
        The reduced Dynes parameter (gamma / delta0).
    bcs: boolean (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    delta_kwargs: optional keyword arguments
        Optional arguments to send to
        superconductivity.gap_functions.reduced_delta_bcs()
    Returns
    -------
    nqpr : float
        The reduced quasiparticle density.
    """
    if g == 0:
        return reduced_density_bcs(t, bcs=bcs, **delta_kwargs)
    nqpr = np.empty(t.shape)
    dr = reduced_delta_dynes(t, g, **delta_kwargs)
    tcr = reduced_tc_dynes(g)
    zero = (t == 0)
    nqpr[zero] = 0
    nqpr_zero = nqpr[~zero]
    for ii, tii in enumerate(t[~zero]):
        nqpr_zero[ii] = it.quad(_integrand_dynes, 0, np.inf, args=(tii, dr[~zero][ii], bcs, g, tcr))[0]
    nqpr[~zero] = nqpr_zero
    return nqpr


def density_dynes(temp, tc, n0, g, bcs=BCS, **delta_kwargs):
    """
    The BCS quasiparticle density calculated either numerically or with the low
    temperature formula.
    Parameters
    ----------
    temp : float or numpy.ndarray
        The temperature in units of Kelvin.
    tc: float
        The transition temperature in Kelvin.
    n0: float
        The single spin density of states in units of states / Joule / volume.
    g: float
        The reduced Dynes parameter (gamma / delta0).
    bcs: boolean (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    delta_kwargs: optional keyword arguments
        Optional arguments to send to
        superconductivity.gap_functions.reduced_delta_bcs()
    Returns
    -------
    nqp : float
        The quasiparticle density in units of number / volume (see n0).
    """
    nqpr = reduced_density_dynes(temp / tc, g, bcs=bcs, **delta_kwargs)
    delta0 = delta_dynes(0, tc, g, bcs=bcs, **delta_kwargs)
    nqp = 4 * n0 * delta0 * nqpr
    return nqp
