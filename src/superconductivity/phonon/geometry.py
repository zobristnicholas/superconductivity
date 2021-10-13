import logging
import numpy as np
import multiprocessing as mp

from superconductivity.utils import initialize_worker, map_async_stoppable
from superconductivity.phonon.acoustic_mismatch import (
    incident_l, incident_sh, incident_sv)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Geometry:
    def __init__(self, width, thickness, density, longitudinal_speed,
                 transverse_speed, membrane=None, diffuse=False,
                 dimensions=3, seed=None):
        """
        Phonon ray tracing in a film with finite width and thickness but
        infinite length. All units are assumed to be SI base units.

        Args:
            width: float
                The width of the film.
            thickness: N array
                The thickness of each layer bottom to top not including the
                substrate.
            density: N + 1 array
                The density of each layer bottom to top including the
                substrate.
            longitudinal_speed: N + 1 array
                The longitudinal speed of sound bottom to top including the
                substrate.
            transverse_speed: N + 1 array
                The transverse speed of sound bottom to top including the
                substrate.
            membrane: N array
                The escape chance out of the side wall from bottom to top not
                including the substrate. If the layer is not a membrane set to
                0.
            diffuse: boolean
                If True, diffuse scattering is assumed for all reflected
                vectors. If False, the reflected angle is computed from the
                acoustic mismatch model (only for interfaces).
            dimensions: integer
                The number of dimensions to do the simulation in. If 2, the y
                dimension is not used (x is along the trace width, z is top
                to bottom, and y is down the trace).
            seed: integer
                A seed for the random number generator.
        """
        self.w = width
        self.t = np.array(thickness, ndmin=1, copy=False)
        self.rho = np.array(density, ndmin=1, copy=False)
        self.cl = np.array(longitudinal_speed, ndmin=1, copy=False)
        self.ct = np.array(transverse_speed, ndmin=1, copy=False)
        self.rng = np.random.default_rng(seed=seed)
        self.diffuse = diffuse
        if dimensions != 2 and dimensions != 3:
            raise ValueError("The 'dimensions' parameter must be 2 or 3.")
        self.dimensions = dimensions

        # derived parameters
        self.z = np.concatenate(([0], np.cumsum(self.t)))  # interfaces
        self.rho = np.concatenate((self.rho, [0]))  # add vacuum
        self.cl = np.concatenate((self.cl, [1]))
        self.ct = np.concatenate((self.ct, [1]))

        self.left_point = np.array([-self.w / 2, 0., 0.])
        self.left_normal = np.array([1., 0., 0.])

        self.right_point = np.array([self.w / 2, 0., 0.])
        self.right_normal = np.array([-1., 0., 0.])

        self.top_normal = np.array([0., 0., -1.])
        self.bottom_normal = np.array([0., 0., 1.])

        if membrane is None:
            membrane = np.zeros_like(self.t)
        self.membrane = membrane

        self.max_iterations = 10000

    def simulate(self, n=100000, parallel=False):
        # Define random positions along x on the top of the geometry.
        positions = self.w * self.rng.random(n) - self.w / 2
        positions = np.array([positions, np.zeros_like(positions),  # x, y, z
                              np.full_like(positions, self.z[-1])]).T

        # Define uniformly distributed directions into the geometry. The
        # normal distribution ensures that there is no angle dependence in
        # the random numbers.
        directions = self.rng.normal(size=(n, 3))
        zero = (directions[:, 2] == 0)  # zero z-velocity
        while zero.any():  # remove any phonons with zero z-velocity
            directions[zero, 2] = self.rng.normal(size=zero.sum())
            zero = (directions[:, 2] == 0)
        directions[directions[:, 2] > 0, 2] *= -1  # flip the up directions
        if self.dimensions == 2:
            directions[:, 1] = 0  # no y velocity if 2D
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        # Randomly select the phonon modes
        mode_types = self.rng.integers(0, high=3, size=n)  # 0, 1, or 2
        modes = np.empty_like(directions)

        # SH waves (perpendicular to direction of motion and top normal).
        shear = (mode_types == 0) | (mode_types == 1)
        modes[shear, :] = np.cross(directions[shear, :], self.top_normal)

        # SV waves.
        sv = (mode_types == 1)
        modes[sv, :] = np.cross(modes[sv, :], directions[sv, :])

        # Longitudinal waves (direction of motion)
        modes[mode_types == 2, :] = directions[mode_types == 2, :]
        modes /= np.linalg.norm(modes, axis=1, keepdims=True)

        # Zero modes are created by directions = [0, 0, 1]
        # Set them to x-polarized.
        zero = (modes == 0).all(axis=1)
        modes[zero, 0] = 1

        # Pick rng seeds to make forked processes all different.
        seeds = self.rng.integers(1, 100000, size=mode_types.size)

        # Loop over phonons.
        times = np.empty(n)
        args = zip(positions, directions, modes, seeds)
        parallel = mp.cpu_count() // 2 if parallel is True else parallel
        if parallel:  # calculate the escape times in parallel
            ctx = mp.get_context("fork")
            with ctx.Pool(parallel, initializer=initialize_worker) as pool:
                results = map_async_stoppable(pool, self.escape, args)
                # wait for the results
                try:
                    log.info("Waiting for results from parallel computation")
                    results.wait()
                except KeyboardInterrupt as error:
                    log.error("Keyboard Interrupt encountered: retrieving "
                              "computed fits before exiting")
                    pool.terminate()
                    pool.join()
                    raise error
                finally:
                    for index, result in enumerate(results.get()):
                        times[index] = result
                    log.info("Retrieved results from parallel computation")
        else:  # calculate the escape times serially
            for index, arg in enumerate(args):
                times[index] = self.escape(*arg)
                log.info(f"Phonon {index} escaped in {times[index]}")
        return times

    def escape(self, position, direction, mode, seed, plot=False):
        init = [position, direction, mode, seed]
        rng = np.random.default_rng(seed=seed)
        total_time = 0
        i = 0
        stop = False
        x, z, axes = [], [], None  # for plotting
        while i <= self.max_iterations:
            if plot:  # for debugging
                # Defer import since matplotlib is not a package dependency.
                from matplotlib import pyplot as plt
                x.append(position[0])
                z.append(position[2])
                if i == 0:  # draw the borders
                    _, axes = plt.subplots()
                    for zi in self.z:
                        axes.plot([-self.w / 2, self.w / 2], [zi, zi], 'k--')
                    z_max = self.z.max()
                    axes.plot([self.w / 2, self.w / 2], [0, z_max], 'k--')
                    axes.plot([-self.w / 2, -self.w / 2], [0, z_max], 'k--')
                elif stop or i == self.max_iterations - 1:
                    axes.plot(x, z)
                    axes.plot([x[0], x[-1]], [z[0], z[-1]], linestyle='none',
                              marker='o')
                    plt.show()

            # Stop when the phonon enters the substrate.
            if stop:
                break

            # Get the next position, direction, mode, and time.
            position, direction, mode, time, stop = self.propagate(
                position, direction, mode, rng)
            total_time += time

            # Increment the iteration counter.
            i += 1

        # Warn if we iterated more than expected and had to stop.
        if i > self.max_iterations:
            with np.printoptions(precision=np.finfo(np.float64).precision + 1,
                                 floatmode='maxprec_equal'):
                log.warning("The maximum number of iterations was achieved "
                            "for these initial parameters: "
                            f"{init[0]}, {init[1]}, {init[2]}, {init[3]}.")

        return total_time

    def propagate(self, position, direction, mode, rng=None):
        down = (direction[2] <= 0)  # phonon traveling down
        left = (direction[0] <= 0)  # phonon traveling left
        if down:
            level = np.nonzero(position[2] <= self.z)[0][0]
            horizontal_point = np.array([0, 0, self.z[level - 1]])
            plane_normal = self.bottom_normal
            rho = [self.rho[level], self.rho[level - 1]]
            ct = [self.ct[level], self.ct[level - 1]]
            cl = [self.cl[level], self.cl[level - 1]]
        else:
            level = np.nonzero(position[2] < self.z)[0][0]
            horizontal_point = np.array([0, 0, self.z[level]])
            plane_normal = self.top_normal
            rho = [self.rho[level], self.rho[level + 1]]
            ct = [self.ct[level], self.ct[level + 1]]
            cl = [self.cl[level], self.cl[level + 1]]

        # Find where the phonon intersects with the vertical wall or
        # horizontal interface.
        new_position = self.intersection(
            position, direction, horizontal_point, plane_normal)
        if not (-self.w / 2 <= new_position[0] <= self.w / 2):
            # The phonon did not hit a horizontal interface so we compute
            # the intersection with the vertical wall.
            plane_normal = self.left_normal if left else self.right_normal
            vertical_point = self.left_point if left else self.right_point
            new_position = self.intersection(
                position, direction, vertical_point, plane_normal)
            # chance of leaving if level extends past the trace width
            if self.membrane[level - 1]:
                if rng.random() < self.membrane[level - 1]:
                    longitudinal = (direction == mode).all()
                    speed = self.cl[level] if longitudinal else self.ct[level]
                    time = np.linalg.norm(position - new_position) / speed
                    stop = True  # exit the material
                    return new_position, direction, mode, time, stop
            rho[1] = 0  # override material2 with vacuum
        else:
            # ensure the z component is identically one of the interfaces
            # to avoid rounding errors.
            new_position[2] = self.z[
                np.argmin(np.abs(new_position[2] - self.z))]

        # Compute the new direction and mode after interacting with the wall
        # or interface.
        new_direction, new_mode = self.interface(
            direction, mode, plane_normal, rho, ct, cl, rng)

        # Calculate the time spent between bounces.
        longitudinal = (direction == mode).all()
        speed = self.cl[level] if longitudinal else self.ct[level]
        time = np.linalg.norm(position - new_position) / speed
        stop = (new_position[2] == 0 and new_direction[2] < 0)
        return new_position, new_direction, new_mode, time, stop

    @staticmethod
    def intersection(position, direction, plane_point, plane_normal):
        """
        Compute the intersection between a line and a plane.

        Args:
            position: numpy.ndarray size 3
                A point on the line.
            direction: numpy.ndarray size 3
                The direction of the line.
            plane_point: numpy.ndarray size 3
                A point on the plane.
            plane_normal: numpy.ndarray size 3
                The normal vector to the plane.
        """
        dot = plane_normal.T @ direction
        if dot == 0:
            with np.errstate(invalid='ignore'):
                intersection = plane_point + np.nan_to_num(
                    direction * np.inf, nan=0.0, posinf=np.inf, neginf=-np.inf)
            return intersection
        w = position - plane_point
        fac = -(plane_normal.T @ w) / dot
        return position + fac * direction

    def interface(self, direction, mode, plane_normal, rho, ct, cl, rng=None):
        # Set the random number generator
        if rng is None:
            rng = np.random.default_rng()

        # Determine the incident angle
        incident_angle = np.arccos(np.dot(direction, -plane_normal))

        # Compute the reflection and transmission probabilities and angles
        if (mode == direction).all():  # longitudinal mode
            r = incident_l(incident_angle, rho, ct, cl)
        else:  # in a superposition of SV and SH
            sh_dir = np.cross(direction, plane_normal)  # direction of SH waves
            if (sh_dir == 0).all():
                sh_dir = mode  # mode is SH
            sh_dir /= np.linalg.norm(sh_dir)
            p_sh = np.dot(sh_dir, mode)**2
            p_sv = max(0., 1 - p_sh)  # max for rounding errors
            incident_mode_type = rng.choice([0, 1], p=[p_sh, p_sv])
            if incident_mode_type == 0:
                r = incident_sh(incident_angle, rho, ct)
            else:
                r = incident_sv(incident_angle, rho, ct, cl)

        # Pick a resulting mode
        options = [[2, False, r["L"]["T"]["angle"]],
                   [2, True, r["L"]["R"]["angle"]],
                   [1, False, r["SV"]["T"]["angle"]],
                   [1, True, r["SV"]["R"]["angle"]],
                   [0, False, r["SH"]["T"]["angle"]],
                   [0, True, r["SH"]["R"]["angle"]]]
        prob = [r["L"]["T"]["P"], r["L"]["R"]["P"],
                r["SV"]["T"]["P"], r["SV"]["R"]["P"],
                r["SH"]["T"]["P"], r["SH"]["R"]["P"]]
        new_mode_type, reflected, angle = rng.choice(options, p=prob)
        if reflected and (rho[1] == 0 or self.diffuse):
            # Randomize the reflected angle due to surface roughness if
            # the transmission interface is vacuum or we are assuming
            # diffuse reflection everywhere.
            # This is what they do in DeVisser et al. 2021
            # (doi:10.1103/PhysRevApplied.16.034051)
            new_direction = rng.normal(size=3)  # angle independent
            z = np.dot(new_direction, plane_normal)
            # Retry if we get a zero out of plane component
            while z == 0:
                new_direction = rng.normal(size=3)
                z = np.dot(new_direction, plane_normal)
            if self.dimensions == 2:
                new_direction[1] = 0  # remove y-component of the new direction
            # Ensure the result is a reflection.
            if z < 0:
                new_direction -= 2 * z * plane_normal
        else:
            # Determine the new direction by scaling the in plane component of
            # the direction vector and keeping the normal component the same.
            z = np.dot(direction, plane_normal) * plane_normal
            xy = direction - z
            ratio = np.tan(angle)
            z_mag = np.sqrt(np.dot(z, z))
            xy_mag = np.sqrt(np.dot(xy, xy)) if ratio != 0 else 1
            xy *= np.abs(ratio * z_mag / xy_mag)
            if reflected:  # specular reflection
                new_direction = xy - z
            else:
                new_direction = xy + z
        new_direction /= np.linalg.norm(new_direction)

        # Determine the new mode
        if new_mode_type == 2:
            new_mode = new_direction
        else:
            new_mode = np.cross(new_direction, plane_normal)  # SH
            if (new_mode == 0).all():  # angle = 0
                new_mode = mode  # SV or SH
            elif new_mode_type == 1:
                new_mode = np.cross(new_direction, new_mode)  # SV
        new_mode /= np.linalg.norm(new_mode)
        return new_direction, new_mode
