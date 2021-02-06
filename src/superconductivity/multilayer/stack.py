import copy
import logging
import numpy as np
import multiprocessing as mp
from scipy.constants import e, k, hbar
from scipy.interpolate import PchipInterpolator

from superconductivity.multilayer.superconductor import Superconductor
from superconductivity.multilayer.usadel import solve_imaginary, solve_real
from superconductivity.utils import (cast_to_list, setup_plot,
                                     finalize_plot, get_scale, raise_bvp)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Stack:
    """
    A collection of materials and boundary resistances which create a
    one dimensional superconducting stack.

    Args:
        layers: iterable of Superconductor or Metal objects
            The different layers of the stack from bottom to top. The
            layers are copied and the originals will not be modified.
        boundaries: iterable of floats
            The boundary resistances in units of Ohm m^2 at each
            interface between layers. There should be one less than the
            number of layers.
    """
    RTOL = 1e-8  # relative convergence tolerance
    MAX_ITERATIONS = 100  # maximum number of iterations to converge
    SPEEDUP = 10  # every <SPEEDUP> iteration is a Steffensen iteration
    THRESHOLD = 1e-3  # DOS threshold for determining the gap energy
    THREADS = True  # Use <THREADS> threads. True uses the maximum available.

    def __init__(self, layers, boundaries):
        layers = cast_to_list(layers)
        boundaries = cast_to_list(boundaries)
        if len(boundaries) != len(layers) - 1:
            raise ValueError("There must be a boundary term for each layer "
                             "interface (one less than the number of layers).")
        self.layers = [copy.deepcopy(m) for m in layers]  # from bottom to top
        self.boundaries = boundaries
        self.tc = None  # The stack's Tc is unknown until update_tc() is called

        # Update the energy scale with which to normalize the calculations.
        self._update_scale()

        # Set up the default grid for the calculations.
        self._update_grid()

        # Initialize all of the layers in their bulk state
        for layer in self.layers:
            layer.initialize_bulk()

        # Update the material dependent arguments to the BVP solver
        self._update_args()

    @property
    def d(self):
        """The total thickness of the stack."""
        return sum([layer.d for layer in self.layers])

    @property
    def order(self):
        """The order parameter for the whole stack at z."""
        return np.concatenate([m.order for m in self.layers])

    def add_top(self, layer, boundary):
        """
        Add a layer to the top of the stack. The boundary term should be
        referenced to the resistivity and coherence length of the layer
        below. Using this method will reset the solution grid to the
        default settings.
        """
        self.layers.append(layer)
        self.boundaries.append(boundary)
        self._update_scale()
        self._update_grid()
        self._update_args()

    def add_bottom(self, layer, boundary):
        """
        Add a layer to the bottom of the stack. The boundary term should
        be referenced to the resistivity and coherence length of the
        layer being added. Using this method will reset the solution
        grid to the default settings.
        """
        self.layers.insert(0, layer)
        self.boundaries.insert(0, boundary)
        self._update_scale()
        self._update_grid()
        self._update_args()

    def update(self):
        """
        Run all of the update routines to make the all of the class
        attributes consistent with the geometry. Note: this will likely
        compute more than you need, so if computation time is important,
        run only the routines required.
        """
        self.update_tc()  # approximate the transition temperature
        self.update_dos()  # computes the density of states
        self.update_gap()  # compute the gap energy

    def update_tc(self):
        pass

    def update_dos(self):
        """
        Update all of the attributes required to compute the density of
        states. This may be more work than required if the order
        parameter is already up to date.
        """
        self.update_order()
        self.update_theta()

    def update_order(self):
        """
        Update the order parameter for the entire stack by solving the
        Usadel diffusion equation and self-consistency equation at the
        Matsubara energies.
        """
        log.info("Computing the order parameter for a stack.")
        # Initialize all of the parameters to their bulk values
        temperatures = []
        for layer in self.layers:
            temperatures.append(layer.t)

        # Enforce a constant temperature
        if not all([temperatures[0] == t for t in temperatures]):
            raise ValueError("All materials must have the same temperature.")

        # Determine the maximum and minimum Matsubara number
        nc = [m.nc for m in self.layers]
        nmax = max(nc)
        nmin = min(nc)

        # Initialize the starting guess and Matsubara energies
        wn = (2 * np.arange(0, nmax + 1) + 1) * np.pi * k * self.layers[0].t
        y_guess = np.zeros((2 * len(self.layers), nmin))

        # Initialize the iteration history.
        order1 = self.order / self.scale
        order2 = order1
        order3 = order2

        # Initialize the while loop termination parameters.
        r = np.inf
        i = 0

        # Iterate between solving for the order and pair angle until the
        # order converges or we run out of iterations.
        while r > self.RTOL and i < self.MAX_ITERATIONS:
            # Update the iteration counter.
            i += 1

            # Perform a Steffensen's iteration to boost convergence speed.
            if self.SPEEDUP and (i % self.SPEEDUP == 0):
                order = order3 - (order2 - order3)**2 / (order1 - 2 * order2
                                                         + order3)

                # Update the order with the boosted value.
                for ii, layer in enumerate(self.layers):
                    start = self.interfaces[ii]
                    stop = self.interfaces[ii + 1]
                    layer.order = order[start:stop] * self.scale

            # Get the order parameter by the solving for the pair angle first.
            else:
                # Make the guess from the last loop's pair angle.
                y_guess[::2, :] = np.array([m.mtheta.mean(axis=1)[:nmin]
                                            for m in self.layers])

                # Solve the diffusion equation at the Matsubara energies.
                theta, info = solve_imaginary(
                    wn / self.scale, self.z, y_guess, self.order / self.scale,
                    self.boundaries, self.interfaces, self.RTOL,
                    self._get_threads(),  **self.kwargs)
                raise_bvp(info)

                # Collect the results into the different layer objects.
                for ii in range(nmax + 1):
                    for iii, layer in enumerate(self.layers):
                        if ii < layer.mtheta.shape[0]:
                            start = self.interfaces[iii]
                            stop = self.interfaces[iii + 1]
                            layer.mtheta[ii, :] = theta[ii, start:stop]

                # Update the order parameter using the new pair angle.
                for layer in self.layers:
                    layer.update_order()

            # Save iteration history and evaluate convergence criteria.
            order3 = order2
            order2 = order1
            order1 = self.order / self.scale
            r = np.max(np.abs(order1 - order2) / (np.abs(order1) + 1))
            log.debug("Iteration: {:d} :: R: {:g}".format(i, r))

        log.info("Order parameter computed.")

    def update_theta(self):
        """
        Update the pair angle for the entire stack based on the stack's
        order parameter by solving the Usadel equation. Run
        update_order() first if the order parameter is not up to date.
        """
        log.info("Computing the pair angle for a stack.")
        # Initialize the guess.
        y_guess = np.zeros((2 * len(self.layers), self.e.size), dtype=complex)
        y_guess[::2, :] = np.pi / 4 + 1j * np.pi / 4

        # Solve the diffusion equation at the requested energies.
        theta, info = solve_real(
            self.e / self.scale, self.z, y_guess, self.order / self.scale,
            self.boundaries, self.interfaces, self.RTOL,
            self._get_threads(), **self.kwargs)
        raise_bvp(info)

        # Collect the results into the different layer objects.
        for i in range(self.e.size):
            for ii, layer in enumerate(self.layers):
                start = self.interfaces[ii]
                stop = self.interfaces[ii + 1]
                layer.theta[i, :] = theta[i, start:stop]

        log.info("Pair angle computed.")

    def update_gap(self):
        pass

    def plot(self, axes_list=None, title=False, title_kwargs=None,
             tick_kwargs=None, tighten=False, order_kwargs=None,
             dos_kwargs=None, **kwargs):
        # Setup the axes.
        if axes_list is None:
            from matplotlib import pyplot as plt
            figure, axes_list = plt.subplots(nrows=2, figsize=[6.4, 4.8 * 2])
        else:
            figure = axes_list[0].figure

        # Plot the order parameter.
        kw = {}  # could add different default settings here
        if order_kwargs is not None:
            kw.update(order_kwargs)
        kw.update(kwargs)

        # Plot the density of states.
        self.plot_order(axes=axes_list[0], **kw)
        kw = {}  # could add different default settings here
        if dos_kwargs is not None:
            kw.update(order_kwargs)
        kw.update(kwargs)
        self.plot_dos(axes=axes_list[1], **kw)

        # Finalize the axes.
        if title:
            kw = {}
            if title_kwargs is not None:
                kw.update(title_kwargs)
            figure.suptitle(title, **kw)
        if tick_kwargs is not None:
            for axes in axes_list:
                axes.tick_params(**tick_kwargs)
        if tighten:
            figure.tight_layout()
        return axes_list

    def plot_order(self, axes=None, energy_scale='meV', title=False,
                   title_kwargs=None, legend=False, legend_kwargs=None,
                   tick_kwargs=None, tighten=False):
        # Set up the energy scale.
        scale = get_scale(energy_scale)

        # Setup the axes.
        _, axes = setup_plot(axes=axes)

        # Loop over the layers.
        for i, layer in enumerate(self.layers):
            # Plot the Pchip interpolation.
            start = self.interfaces[i]
            stop = self.interfaces[i + 1]
            interp = PchipInterpolator(self.z[start:stop],
                                       self.order[start:stop])
            z = np.linspace(self.z[start], self.z[stop - 1],
                            100 * (stop - start))
            line, = axes.plot(z * 1e9, interp(z) / e * scale,
                              label="layer {}".format(i + 1),  linestyle='-')

            # Plot the bulk values.
            if isinstance(layer, Superconductor):
                axes.plot(z * 1e9, np.full(z.shape, layer.delta0 / e * scale),
                          label="bulk {}".format(i + 1),
                          color=line.get_color(), linestyle='--')
            else:
                axes.plot(z * 1e9, np.zeros(z.shape),
                          label="bulk {}".format(i + 1),
                          color=line.get_color(), linestyle='--')

        # Plot the simulation points.
        axes.plot(self.z * 1e9, self.order / e * scale, linestyle='none',
                  marker='o', markersize=3, color='k')

        # Make the axes labels.
        axes.set_ylabel(f"order parameter [{energy_scale}]")
        axes.set_xlabel("position [nm]")

        # Finalize the plot.
        finalize_plot(axes=axes, title=title, title_kwargs=title_kwargs,
                      legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def plot_dos(self, axes=None, location='edges', fix_color=False,
                 energy_scale='meV', title=False, title_kwargs=None,
                 legend=False, legend_kwargs=None, tick_kwargs=None,
                 tighten=False):

        # Set up the energy scale.
        scale = get_scale(energy_scale)

        # Setup the axes.
        _, axes = setup_plot(axes=axes)

        # Parse the location input if it's a string
        n = len(self.layers)
        if isinstance(location, str):
            if location.lower() == 'edges':
                location = [['b', 't'] for _ in range(n)]
            elif location.lower() == 'centers':
                location = [['c'] for _ in range(n)]
            elif location.lower() == 'mixed':
                location = ['b'] + [['c'] for _ in range(n - 2)] + ['t']
            else:
                raise ValueError(f"'{location}' is not a valid location.")

        # Loop over the layers.
        line = None
        energy = self.e / e * scale
        for i, layer in enumerate(self.layers):
            # Reset the line if we are cycling the line color.
            if not fix_color:
                line = None

            # Loop over the locations in the layer
            for loc in location[i]:
                # Plot the bottom of the layer
                if loc.lower().startswith('b'):
                    line, = axes.plot(energy,
                                      np.real(np.cos(layer.theta[:, 0])),
                                      label="layer {}: bottom".format(i + 1),
                                      linestyle='-',
                                      color=line.get_color()
                                      if line is not None else None)

                # Plot the center of the layer
                elif loc.lower().startswith('c'):
                    # Numpy not catching stray warning as of version 1.19.2
                    with np.errstate(invalid='ignore'):
                        pchip = PchipInterpolator(layer.z, layer.theta, axis=1)
                    theta = pchip(layer.z.mean())
                    line, = axes.plot(energy,
                                      np.real(np.cos(theta)),
                                      label="layer {}: center".format(i + 1),
                                      linestyle='--',
                                      color=line.get_color()
                                      if line is not None else None)

                # Plot the top of the layer
                elif loc.lower().startswith('t'):
                    line, = axes.plot(energy,
                                      np.real(np.cos(layer.theta[:, -1])),
                                      label="layer {}: top".format(i + 1),
                                      linestyle='-.',
                                      color=line.get_color()
                                      if line is not None else None)
                else:
                    raise ValueError(f"'{loc}' is not a valid location.")

        # Make the axes labels.
        axes.set_ylabel("normalized density of states")
        axes.set_xlabel(f"energy [{energy_scale}]")

        # Finalize the plot.
        finalize_plot(axes=axes, title=title, title_kwargs=title_kwargs,
                      legend=legend, legend_kwargs=legend_kwargs,
                      tick_kwargs=tick_kwargs, tighten=tighten)
        return axes

    def _update_scale(self):
        tcs = [m.tc for m in self.layers if isinstance(m, Superconductor)]
        self.scale = np.pi * k * max(tcs) if tcs else 1

    def _update_grid(self):
        # Set the z grid.
        start, stop = [], []
        for layer in self.layers:
            start.append(sum(stop[-1:]))
            stop.append(layer.d + sum(stop[-1:]))
            layer.z = np.linspace(start[-1], stop[-1], layer.Z_GRID)
        self.z = np.concatenate([m.z for m in self.layers])

        # Set the interface indices for the z grid.
        self.interfaces = np.nonzero(self.z == np.roll(self.z, 1))[0]
        self.interfaces = np.append(0, self.interfaces)
        self.interfaces = np.append(self.interfaces, len(self.z))

        if self.interfaces.size != len(self.layers) + 1:
            raise AttributeError("The position grid does not have the "
                                 "correct number of layers.")

        # Set the energy grid.
        if any([m.E_GRID != self.layers[0].E_GRID for m in self.layers]):
            raise AttributeError("All layers must have the same E_GRID "
                                 "constant.")
        energies = np.linspace(0, 2 * self.scale, self.layers[0].E_GRID)
        self.e = energies
        for layer in self.layers:
            layer.e = self.e

    def _update_args(self):
        # update arguments to the bvp solver that only change when a new
        # material is added to the stack.
        d = np.array([layer.d for layer in self.layers])
        z = np.empty(self.z.shape)
        for i in range(len(self.layers)):
            start = self.interfaces[i]
            stop = self.interfaces[i + 1]
            if i % 2 == 0:
                z[start:stop] = (self.z[start:stop] - np.sum(d[:i])) / d[i]
            else:
                z[start:stop] = 1 - (self.z[start:stop] - np.sum(d[:i])) / d[i]
        # Round to avoid floating point errors.
        _, indices = np.unique(z.round(decimals=10), return_index=True)
        z_guess = z[indices]
        self.kwargs = {"d": d,
                       "rho": np.array([m.rho for m in self.layers]),
                       "z_scale": np.array([(2 * self.scale * m.d**2
                                             / (hbar * m.dc))
                                            for m in self.layers]),
                       "z_guess": z_guess}

    def _get_threads(self):
        if self.THREADS is True:
            return mp.cpu_count()
        elif self.THREADS is False:
            return 1
        else:
            return self.THREADS
