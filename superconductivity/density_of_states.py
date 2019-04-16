import numpy as np


def model(en, delta, gamma=0, amplitude=0, ep=0, fwhm=0):
    """
    Compute the density of states for a generic BCS superconductor with the
    option of a small Lorentzian distribution of inter-gap states. Functional
    form from Noguchi T. et al. IEEE Trans. Appl. SuperCon., 28, 4, 2018.
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float, complex
        Superconducting gap energy in units of en. An imaginary part signifies
        finite loss at zero temperature.
    gamma: float (optional)
        Dynes parameter for broadening the density of states in units of en.
        Typically not used if delta is complex.
        Dynes et al. Phys. Rev. Lett., 41, 21, 1978.
    amplitude: float (optional)
        Area under the Lorentzian curve for the inter-gap states. If zero
        (default) ep and fwhm are ignored.
    ep: float (optional)
        Energy relative to the fermi energy in units of en for the Lorentzian
        distribution.
    fwhm: float (optional)
        Full-width-half-max energy in units of en for the Lorentzian
        distribution. Cannot be zero if amplitude is not zero.
    Returns
    -------
    dos: float, numpy.ndarray
        density of states as a function of en
    """
    assert not (amplitude != 0 and fwhm == 0), "If the amplitude is not zero, fwhm must not be zero"
    dos = np.abs(np.real((en - 1j * gamma) / np.sqrt((en - 1j * gamma)**2 - delta**2)))
    if amplitude != 0 and fwhm != 0:
        dos += 2 * amplitude * fwhm / np.pi / (4 * (en - ep)**2 + fwhm**2)
    return dos
