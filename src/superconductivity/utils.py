import signal
import numpy as np
from collections import OrderedDict

BCS = np.pi / np.exp(np.euler_gamma)


def raise_bvp(info):
    if info == 0:
        return  # no error
    prefix = "BVP_SOLVER error: "
    if info == -1:
        raise RuntimeError(prefix + 'Number of subintervals needed for new ' 
                           'mesh would exceed current allowed maximum.')
    elif info == 1:
        raise RuntimeError(prefix + 'The Newton iteration failed because the '
                           'maximum number of Newton iterations was '
                           'exceeded.')
    elif info == 2:
        raise RuntimeError(prefix + 'The Newton iteration failed because a '
                           'singular Newton matrix was encountered.')
    elif info == 3:
        raise RuntimeError(prefix + 'The Newton iteration has been terminated '
                           'because it is not converging at a satisfactory '
                           'rate.')
    else:
        raise RuntimeError(prefix + f"Unknown info parameter: '{info}'")


def get_scale(energy_scale):
    units1 = ['neV', 'µeV', 'meV', 'eV', 'keV', 'MeV', 'GeV']  # greek
    units2 = ['neV', 'ueV', 'meV', 'eV', 'keV', 'MeV', 'GeV']  # no greek
    if energy_scale in units1:
        units = units1
    elif energy_scale in units2:
        units = units2
    else:
        raise ValueError(f"'energy_scale' must be in {units2}.")
    scale = 1
    for i, unit in enumerate(units):
        if unit == energy_scale:
            scale = 10 ** (-3 * (i - 3))
            break
    return scale


def setup_plot(axes=None):
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    return figure, axes


def finalize_plot(axes, title=False, title_kwargs=None, legend=False,
                  legend_kwargs=None, tick_kwargs=None, tighten=False):
    # make the legend
    if legend:
        kwargs = {}  # default settings
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        axes.legend(**kwargs)
    # make the title
    if title:
        kwargs = {}
        if title_kwargs is not None:
            kwargs.update(title_kwargs)
        axes.set_title(title, **kwargs)
    # modify ticks
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    # tighten the figure
    if tighten:
        axes.figure.tight_layout()


def cast_to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


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


def initialize_worker():
    """Initialize multiprocessing.pool worker to ignore keyboard interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def map_async_stoppable(pool, func, iterable, callback=None):
    results = MapResult()
    for item in iterable:
        results.append(pool.apply_async(func, cast_to_list(item),
                                        callback=callback))
    return results


class MapResult(list):
    def get(self, *args, **kwargs):
        results = []
        for r in self:
            if r.ready():
                results.append(r.get(*args, **kwargs))
            else:
                results.append(None)
        return results

    def wait(self, timeout=None):
        for r in self:
            r.wait(timeout)
