import numpy as np
import numba as nb
from superconductivity.density_of_states import usadel_pairing_angle


def dop_bcs(en, delta, real=True):
    """
    Compute the density of pairs for a generic BCS superconductor.
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float
        Superconducting gap energy in units of en.
    real: boolean (optional)
        If False, the imaginary part of the complex valued function is
        returned. If True, the real part of the complex valued function
        is returned. The real part is the density of pairs. The default is
        True.
    Returns
    -------
    dop: numpy.ndarray
        density of pairs as a function of en
    """
    en = np.atleast_1d(en)
    dop = np.empty(en.shape, dtype=np.complex)
    _dop(dop, en, delta, 0, real=real)
    return dop.real if real else dop.imag


def dop_dynes(en, delta, gamma, real=True):
    """
    Compute the density of pairs for a Dynes superconductor. Functional
    form from Herman et al. Phys. Rev. B, 96, 1, 2017.
    (doi:10.1103/PhysRevB.96.014509)
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float
        Superconducting gap energy in units of en.
    gamma: float (optional)
        Dynes parameter for broadening the density of states in units of en.
    real: boolean (optional)
        If False, the imaginary part of the complex valued function is
        returned. If True, the real part of the complex valued function
        is returned. The real part is the density of pairs. The default is
        True.
    Returns
    -------
    dop: numpy.ndarray
        density of pairs as a function of en
    """
    en = np.atleast_1d(en)
    dop = np.empty(en.shape, dtype=np.complex)
    _dop(dop, en, delta, gamma, real=real)
    return dop.real if real else dop.imag


def dop_usadel(en, delta, alpha, real=True):
    """
    Compute the density of pairs for an Usadel superconductor. Functional
    form from Coumou et al. Phys. Rev. B, 88, 18, 2013.
    (doi:10.1103/PhysRevB.88.180505)
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float
        Superconducting gap energy in units of en.
    alpha: float
        The disorder-dependent pair-breaking parameter in units of en.
    real: boolean (optional)
        If False, the imaginary part of the complex valued function is
        returned. If True, the real part of the complex valued function
        is returned. The real part is the density of states. The default is
        True.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    theta = usadel_pairing_angle(en, delta, alpha)
    dop = np.sin(theta)
    return dop.real if real else dop.imag


@nb.njit(cache=True)
def _dop(data, en, delta, gamma, real=True):
    zero = np.sqrt((en + 1j * gamma)**2 - delta**2) == 0
    data[zero & (en > 0)] = np.inf if real else -1j * np.inf
    data[zero & (en <= 0)] = -np.inf if real else -1j * np.inf
    en_c = en[~zero] + 1j * gamma
    data[~zero] = np.sign(en[~zero] + 1j) * delta / np.sqrt(en_c**2 - delta**2)
