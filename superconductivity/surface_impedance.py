import numpy as np
import scipy.constants as sc

from superconductivity import complex_conductivity as cc
from superconductivity.utils import BCS, coerce_arrays, get_fermi_velocity, split_sigma


def extreme_anomalous_limit(temp, freq, delta0, lambda0, vf=None, xi0=None, low_energy=False, bcs=BCS):
    """
    Calculate the surface impedance in the thick film, extreme anomalous limit
    (xi0 >> lambda0 and mfp >> lambda0).
    Parameters
    ----------
    temp: float, iterable of size N
        Temperature in units of Kelvin.
    freq: float, iterable of size N
        Frequency in units of Hz.
    delta0: float, complex
        Superconducting gap energy at 0 Kelvin in units of Joules. An imaginary
        part signifies finite loss at zero temperature.
    lambda0: float
        The london penetration depth in meters
    vf: float (optional)
        The fermi velocity in meters per second. If not specified, xi0 must be
        specified.
    xi0: float (optional)
        The coherence length in meters. If not specified, vf must be specified.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). Can dramatically speed up computation time.
        Default is False.
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity.
    Returns
    -------
    zs: complex, iterable of size N
        the surface impedance in ohms
    """
    # coerce inputs
    temp, freq = coerce_arrays(temp, freq)
    vf = get_fermi_velocity(vf, xi0)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get complex conductivity
    sigma1, sigma2 = split_sigma(cc.value(temp, freq, delta0, low_energy=low_energy, bcs=bcs))
    # compute surface impedance
    prefactor = 1j * sc.mu_0 * np.sqrt(3) * np.pi * freq
    zs = prefactor * (3 * np.pi**2 * freq / (2 * vf * lambda0**2) * (sigma2 + 1j * sigma1))**(-1 / 3)
    return zs


def local_limit(temp, freq, delta0, lambda0, mfp, vf=None, xi0=None, low_energy=False, bcs=BCS):
    """
    Calculate the surface impedance in the thick film, local limit
    (xi0 << lambda0 or mfp << lambda0).
    Parameters
    ----------
    temp: float, iterable of size N
        Temperature in units of Kelvin.
    freq: float, iterable of size N
        Frequency in units of Hz.
    delta0: float, complex
        Superconducting gap energy at 0 Kelvin in units of Joules. An imaginary
        part signifies finite loss at zero temperature.
    lambda0: float
        The london penetration depth in meters
    mfp: float
        The mean free path in meters
    vf: float (optional)
        The fermi velocity in meters per second. If not specified, xi0 must be
        specified.
    xi0: float (optional)
        The coherence length in meters. If not specified, vf must be specified.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). Can dramatically speed up computation time.
        Default is False.
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity.
    Returns
    -------
    zs: complex, iterable of size N
        the surface impedance in ohms
    """
    # coerce inputs
    temp, freq = coerce_arrays(temp, freq)
    vf = get_fermi_velocity(vf, xi0)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get complex conductivity
    sigma1, sigma2 = split_sigma(cc.value(temp, freq, delta0, low_energy=low_energy, bcs=bcs))
    # compute surface impedance
    zs = 2j * pi * sc.mu_0 * freq * np.sqrt(2 * np.pi * freq * mfp * (sigma2 + 1j * sigma1) / (vf * lambda0**2))
    return zs


def thin_film_limit(temp, freq, delta0, d, lambda0, mfp, vf=None, xi0=None, low_energy=False, bcs=BCS):
    """
    Calculate the surface impedance in the thick film, local limit
    (xi0 << lambda0 or mfp << lambda0).
    Parameters
    ----------
    temp: float, iterable of size N
        Temperature in units of Kelvin.
    freq: float, iterable of size N
        Frequency in units of Hz.
    delta0: float, complex
        Superconducting gap energy at 0 Kelvin in units of Joules. An imaginary
        part signifies finite loss at zero temperature.
    d: float
        The film thickness in meters
    lambda0: float
        The london penetration depth in meters
    mfp: float
        The mean free path in meters
    vf: float (optional)
        The fermi velocity in meters per second. If not specified, xi0 must be
        specified.
    xi0: float (optional)
        The coherence length in meters. If not specified, vf must be specified.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). Can dramatically speed up computation time.
        Default is False.
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity.
    Returns
    -------
    zs: complex, iterable of size N
        the surface impedance in ohms
    """
    # coerce inputs
    temp, freq = coerce_arrays(temp, freq)
    vf = get_fermi_velocity(vf, xi0)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get complex conductivity
    sigma = cc.value(temp, freq, delta0, low_energy=low_energy, bcs=bcs)
    zs = 1 / (d * sigma * mfp / (sc.mu_0 * vf * lambda0**2))
    return zs


def numeric(temp, freq, delta0, lambda0, mfp, vf=None, xi0=None):
    temp, freq = coerce_arrays(temp, freq)
    vf = get_fermi_velocity(vf, xi0)
    assert (temp > 0).all(), "Temperature must be >= 0."

    delta1 = np.real(delta0)
    delta2 = np.imag(delta1)
    raise NotImplementedError
