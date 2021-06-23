import numpy as np
import numba as nb
from scipy.optimize import root


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
    dos = np.empty(en.shape, dtype=np.complex)
    _dos(dos, en, delta, gamma, real=real)
    return dos.real if real else dos.imag


def dos_usadel(en, delta, alpha, real=True):
    """
    Compute the density of states for an Usadel superconductor. Functional
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
    theta = usadel_pair_angle(en, delta, alpha)
    dos = np.cos(theta)
    return dos.real if real else dos.imag


def usadel_pair_angle(en, delta, alpha):
    shape = en.shape
    en = en.ravel() / delta
    alpha = alpha / delta
    x0 = np.full(en.size * 2, np.pi / 4)
    if np.iscomplex(en).any():
        x0[en.size:] = 0
    sol = root(_usadel, x0, args=(en, 1, alpha), method='krylov')
    if sol.x[en.size:].any():
        theta = sol.x[:en.size] + 1j * sol.x[en.size:]
        # fix small numerical errors that aren't physical
        bad = (theta.real > np.pi / 2)
        if bad.any():
            x0 = np.full(en[bad].size * 2, np.pi / 4)
            sol = root(_usadel, x0, args=(en[bad], 1, alpha), method='krylov')
            theta[bad] = sol.x[:en[bad].size] + 1j * sol.x[en[bad].size:]
        still_bad = (theta.real > np.pi / 2)
        theta[still_bad] = np.pi / 2 + 1j * theta[still_bad].imag
    else:
        theta = sol.x[:en.size]
    return theta.reshape(shape)


# @nb.njit(cache=True)
def _usadel(theta, en, delta, alpha):
    th = theta[:int(theta.size // 2)] + 1j * theta[int(theta.size // 2):]
    fc = (1j * en * np.sin(th) + delta * np.cos(th)
          - alpha * np.sin(th) * np.cos(th))
    f = np.concatenate((fc.real, fc.imag))
    return f


@nb.njit(cache=True)
def _dos(data, en, delta, gamma, real=True):
    zero = np.sqrt((en + 1j * gamma)**2 - delta**2) == 0
    data[zero & (en > 0)] = np.inf if real else -1j * np.inf
    data[zero & (en <= 0)] = np.inf if real else 1j * np.inf
    en_c = en[~zero] + 1j * gamma
    data[~zero] = np.sign(en[~zero] + 1j) * en_c / np.sqrt(en_c**2 - delta**2)
