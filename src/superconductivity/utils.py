import numpy as np
from collections import OrderedDict

BCS = np.pi / np.exp(np.euler_gamma)


def coerce_arrays(array1, array2):
    """Make arrays the same shape and size if one array has size 1."""
    array1 = np.atleast_1d(array1)
    array2 = np.atleast_1d(array2)
    if array2.size == 1 and array1.size != 1:
        array2 = np.full(array1.shape, array2)
    elif array1.size == 1 and array2.size != 1:
        array1 = np.full(array2.shape, array1)
    elif array1.shape != array2.shape:
        raise ValueError("Incompatible array sizes")
    return array1, array2


def get_fermi_velocity(vf, xi0):
    """
    Parse inputs to allow specifying either the fermi velocity or coherence
    length.
    """
    if vf is None and xi0 is not None:
        vf = np.pi * np.real(delta0) * xi0 / sc.hbar
    elif vf is None:
        raise ValueError("One of vf or xi0 must be specified")
    return vf


def split_sigma(sigma):
    """
    Split the complex conductivity into sigma1, sigma2.
    This is mainly here so that minus signs don't get messed up.
    """
    return sigma.real, -sigma.imag


def combine_sigma(sigma1, sigma2):
    """
    Combine the complex conductivity into one complex number. This is mainly
    here so that minus signs don't get messed up.
    """
    return sigma1 - 1j * sigma2


class RotatingDict(OrderedDict):
    """Ordered dictionary with a max size."""
    def __init__(self, *args, max_size=10, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        while len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)
