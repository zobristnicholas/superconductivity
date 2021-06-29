import os
import logging
import numpy as np
import numba as nb
from scipy.optimize import root

from superconductivity.utils import initialize_worker, map_async_stoppable

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
        is returned. The real part is the density of states. The default
        is True.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    en = np.atleast_1d(en)
    dos = np.empty(en.shape, dtype=np.complex)
    _dos(dos, en, delta, 0, real=real)
    return dos.real if real else dos.imag


def dos_dynes(en, delta, gamma, real=True):
    """
    Compute the density of states for a Dynes superconductor. Functional
    form from Herman et al. Phys. Rev. B, 96, 1, 2017.
    (doi: 10.1103/PhysRevB.96.014509)
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float
        Superconducting gap energy in units of en.
    gamma: float
        Dynes parameter for broadening the density of states in units of
        en.
    real: boolean (optional)
        If False, the imaginary part of the complex valued function is
        returned. If True, the real part of the complex valued function
        is returned. The real part is the density of states. The default
        is True.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    en = np.atleast_1d(en)
    dos = np.empty(en.shape, dtype=np.complex)
    _dos(dos, en, delta, gamma, real=real)
    return dos.real if real else dos.imag


def dos_usadel(en, delta, alpha, real=True):
    """
    Compute the density of states for an Usadel superconductor.
    Functional form from Coumou et al. Phys. Rev. B, 88, 18, 2013.
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
        is returned. The real part is the density of states. The default
        is True.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    theta = usadel_pairing_angle(en, delta, alpha)
    dos = np.cos(theta)
    return dos.real if real else dos.imag


def usadel_pairing_angle(en, delta, alpha):
    """
    Compute the superconducting pairing angle for an Usadel
    superconductor. Functional form from Coumou et al. Phys. Rev. B, 88,
    18, 2013. (doi:10.1103/PhysRevB.88.180505)
        Parameters
    ----------
    en: float or complex, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units. A
        complex energy is interpreted as evaluating the usadel equation
        along the complex axis (at the Matsubara frequencies).
    delta: float
        Superconducting gap energy in units of en.
    alpha: float
        The disorder-dependent pair-breaking parameter in units of en.
    Returns
    -------
    theta: numpy.ndarray
        The superconducting pairing angle as a function of en
    """
    shape = np.array(en).shape
    en = np.array(en).ravel() / delta
    alpha = alpha / delta

    # At the imaginary Matsubara energies the Usadel equation is real.
    if np.iscomplex(en).any():
        # The krylov method is faster than iterating over all of the
        # energies for diagonal systems.
        x0 = np.full(en.size, np.pi / 4)
        theta = root(_usadel, x0, args=(en, 1, alpha), method='krylov').x
    # At real energies the Usadel equation is complex.
    else:
        # Since theta is complex, the system is no longer diagonal and it is
        # faster to loop over each energy.
        def find_root(e, a, guess):
            e = np.atleast_1d(e)
            # 'hybr' method is faster for small problems
            s = root(_usadel, guess, args=(e, 1, a), method='hybr')
            return s.x[0] + 1j * s.x[1]

        x0 = np.full(2, np.pi / 4)
        theta = np.array([find_root(e, alpha, x0) for e in en])
    return theta.reshape(shape)


@nb.njit(cache=True)
def _usadel(theta, en, delta, alpha):
    if theta.size > en.size:
        th = theta[:int(theta.size // 2)] + 1j * theta[int(theta.size // 2):]
    else:
        th = theta + 0j
    fc = (1j * en * np.sin(th) + delta * np.cos(th)
          - alpha * np.sin(th) * np.cos(th))
    f = np.concatenate((fc.real, fc.imag)) if theta.size > en.size else fc.real
    return f


@nb.njit(cache=True)
def _dos(data, en, delta, gamma, real=True):
    zero = np.sqrt((en + 1j * gamma)**2 - delta**2) == 0
    data[zero & (en > 0)] = np.inf if real else -1j * np.inf
    data[zero & (en <= 0)] = np.inf if real else 1j * np.inf
    en_c = en[~zero] + 1j * gamma
    data[~zero] = np.sign(en[~zero] + 1j) * en_c / np.sqrt(en_c**2 - delta**2)
