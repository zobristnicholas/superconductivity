import warnings
import numpy as np
from numpy.lib.scimath import sqrt  # sqrt that doesn't error on negative floats


def dos_bcs(en, delta, real=True):
    """
    Compute the density of states for a generic BCS superconductor.
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float
        Superconducting gap energy in units of en.
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
    en = np.atleast_1d(en)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # divide by zero
        dos = np.sign(en) * en / sqrt(en**2 - delta**2)
    if not real:  # imaginary part has the wrong sign for energies less than zero
        logic = (en < 0)
        dos[logic] = np.conj(dos[logic])
    return dos.real if real else dos.imag


def dos_dynes(en, delta, gamma, real=True):
    """
    Compute the density of states for a Dynes superconductor. Functional
    form from Herman et al. Phys. Rev. B, 96, 1, 2017.
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
        is returned. The real part is the density of states. The default is
        True.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    en = np.atleast_1d(en)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # divide by zero
        dos = np.sign(en) * (en + 1j * gamma) / sqrt((en + 1j * gamma)**2 - delta**2)
    return dos.real if real else dos.imag
