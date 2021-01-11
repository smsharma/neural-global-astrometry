import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow

from theory.units import *
from theory.profiles import Profiles


class DemoSim:
    def __init__(self, theta_x_lims=[-1.6, 1.6], theta_y_lims=[-0.9, 0.9]):
        """
        Class to create animations of astrometric weak lensing. For demo and not science!

        :param theta_x_lims: x-axis coordinate limits [x_min, x_max] in arcsecs
        :param theta_y_lims: y-axis coordinate limits [y_min, y_max] in arcsecs
        """

        self.theta_x_lims = theta_x_lims
        self.theta_y_lims = theta_y_lims

        # Area of sky region
        self.roi_area = (self.theta_x_lims[1] - self.theta_x_lims[0]) * \
                        (self.theta_y_lims[1] - self.theta_y_lims[0])

    def animation(self, pos_l, M_l, R_l, v_l, D_l, n_lens_x=200, n_lens_y=200,  # Lens properties
                  # Source properties
                  n_dens=20, source_pos="random", custom_source_pos=None, arrow_mult=2000,
                  # Figure properties
                  figsize=(16, 9), animate=True, dt=10, interval=10, n_frames=100,
                  # What to plot/animate
                  show_lens=True, show_sources=True, show_orig=False, show_vel_arrows=True,
                  star_kwargs={}, star_orig_kwargs={}, arrow_kwargs={}  # Source plotting properties
                  ):
        """
        :param pos_l: tuple of lens positions, format [[x_1, y_1], [x_2, y_2]...]
        :param M_l: tuple of lens masses
        :param R_l: tuple of lens sizes (Gaussian lens)
        :param v_l: tuple of lens velocities
        :param D_l: tuple of distances of lenses
        :param n_lens_x: number of x-grid points across canvas to plot lenses; default 200
        :param n_lens_y: number of y-grid points across canvas to plot lenses; default 200
        :param n_dens: density of sources (per arcsecs^2); default 20
        :param source_pos: must be one of ["uniform", "random"]; default "random"
        :param custom_source_pos: optional custom positions to put down sources, format [[x_1, y_1], [x_2, y_2]...]
        :param arrow_mult: stretch velocity arrows by this factor; default 2000
        :param figsize: size of canvas, according to matplotlib format; default (16, 9)
        :param animate: whether to animate; default True
        :param dt: cosmological time interval between subsequent animation frames; default 10 [years]
        :param interval: real time interval between subsequent animation frames; default 10 ms
        :param n_frames: number of frames to animate; default 100
        :param show_lens: whether to plot lens; default True
        :param show_sources: whether to plot sources; default True
        :param show_orig: whether to plot unperturbed sources; default False
        :param show_vel_arrows: whether to plot instantaneous velocity vectors; default False
        :param star_kwargs: star plotting options; matplotlib defaults by default
        :param star_orig_kwargs: unperturbed star plotting options; matplotlib defaults by default
        :param arrow_kwargs: arrow plotting options; matplotlib defaults by default
        :return: animation or figure, depending on whether `animation=True` or `animation=False`
        """

        # Set global parameters
        self.arrow_mult = arrow_mult
        self.show_lens = show_lens
        self.show_orig = show_orig
        self.show_vel_arrow = show_vel_arrows
        self.show_sources = show_sources

        self.star_kwargs = star_kwargs
        self.star_orig_kwargs = star_orig_kwargs
        self.arrow_kwargs = arrow_kwargs

        # Get total number of sources in sky region
        self.n_total = np.random.poisson(n_dens * self.roi_area)

        # Set source positions

        # Random positions + custom if specified
        if source_pos == "random":

            # Initial source property array
            self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                         ("theta_y", float, 1),
                                                         ("theta_x_0", float, 1),
                                                         ("theta_y_0", float, 1),
                                                         ("mu", float, 1)])

            if custom_source_pos is None:
                self.sources["theta_x_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total)))
                self.sources["theta_y_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total)))
            else:  # Add custom source positions
                self.n_custom = len(custom_source_pos)
                self.sources["theta_x_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(
                        custom_source_pos[:, 0]))
                self.sources["theta_y_0"] = np.array(
                    list(np.random.uniform(*self.theta_x_lims, self.n_total - self.n_custom)) + list(
                        custom_source_pos[:, 1]))

        # Uniform grid of sources
        elif source_pos == "uniform":
            xy_ratio = (self.theta_y_lims[1] - self.theta_y_lims[0]) / \
                (self.theta_x_lims[1] - self.theta_x_lims[0])
            x_pos = np.linspace(self.theta_x_lims[0], self.theta_x_lims[1], np.round(
                np.sqrt(self.n_total / xy_ratio)))
            y_pos = np.linspace(self.theta_y_lims[0], self.theta_y_lims[1], np.round(
                np.sqrt(self.n_total * xy_ratio)))

            self.n_total = len(np.meshgrid(x_pos, y_pos)[0].flatten())

            # Initialize source property array
            self.sources = np.zeros(self.n_total, dtype=[("theta_x", float, 1),
                                                         ("theta_y", float, 1),
                                                         ("theta_x_0", float, 1),
                                                         ("theta_y_0", float, 1)])

            self.sources["theta_x_0"] = np.meshgrid(x_pos, y_pos)[0].flatten()
            self.sources["theta_y_0"] = np.meshgrid(x_pos, y_pos)[1].flatten()

        assert len(pos_l) == len(v_l) == len(M_l) == len(R_l) == len(D_l), \
            "Lens property arrays must be the same size!"

        # Infer number of lenses
        self.n_lens = len(pos_l)

        # Initialize lens property array
        self.lenses = np.zeros(self.n_lens, dtype=[("theta_x", float, 1),
                                                   ("theta_y", float, 1),
                                                   ("M_0", float, 1),
                                                   ("R_0", float, 1),
                                                   ("D", float, 1),
                                                   ("v_x", float, 1),
                                                   ("v_y", float, 1)])

        # Set initial source positions
        self.sources["theta_x"] = self.sources["theta_x_0"]
        self.sources["theta_y"] = self.sources["theta_y_0"]

        # Set initial lens positions...
        self.lenses["theta_x"] = np.array(pos_l)[:, 0]
        self.lenses["theta_y"] = np.array(pos_l)[:, 1]

        # ... and lens properties
        self.lenses["v_x"] = np.array(v_l)[:, 0]
        self.lenses["v_y"] = np.array(v_l)[:, 1]
        self.lenses["M_0"] = np.array(M_l)
        self.lenses["R_0"] = np.array(R_l)
        self.lenses["D"] = np.array(D_l)

        # Initialize plot
        # fig = plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        self.ax = plt.axes(xlim=self.theta_x_lims, ylim=self.theta_y_lims)
        self.ax.set_facecolor('black')

        if self.show_orig:  # Show original star positions
            self.scatter = self.ax.scatter(
                self.sources["theta_x"], self.sources["theta_y"], **self.star_orig_kwargs)

        if self.show_lens:  # Show lens positions
            self.x_coords = np.linspace(
                self.theta_x_lims[0], self.theta_x_lims[1], n_lens_x)
            self.y_coords = np.linspace(
                self.theta_y_lims[0], self.theta_y_lims[1], n_lens_y)

            im = np.zeros((n_lens_x, n_lens_y))

            for i_lens in range(self.n_lens):
                self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"][i_lens],
                                                       self.y_coords - self.lenses["theta_y"][i_lens])
                r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
                im += \
                    np.transpose(Profiles.MdMdb_Gauss(r_grid, self.lenses["R_0"][i_lens] / self.lenses["D"][i_lens] * radtoasc,
                                                      self.lenses["M_0"][i_lens])[0])

            self.imshow = self.ax.imshow(im, origin='lower', cmap='binary',
                                         extent=[*self.theta_x_lims, *self.theta_y_lims])

        mu_s = np.zeros((self.n_total, 2))
        theta_s = np.zeros((self.n_total, 2))

        # Deflection and proper motion vectors
        for i_lens in range(self.n_lens):
            b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                                  self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens],
                              self.lenses["v_y"][i_lens]])
            for i_source in range(self.n_total):
                mu_s[i_source] += self.mu(b_ary[i_source], vel_l / self.lenses["D"][i_lens],
                                          self.lenses["R_0"][i_lens],
                                          self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])
                theta_s[i_source] += self.theta(b_ary[i_source], self.lenses["R_0"][i_lens],
                                                self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])

        # New source positions including deflection
        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        if self.show_sources:
            self.scatter = self.ax.scatter(
                self.sources["theta_x"], self.sources["theta_y"], **self.star_kwargs)

        if self.show_vel_arrow:
            self.arrows = []

            for i_source in range(self.n_total):
                self.arrows.append(self.ax.add_patch(Arrow(self.sources["theta_x"][i_source],
                                                           self.sources["theta_y"][i_source],
                                                           mu_s[i_source, 0] *
                                                           self.arrow_mult,
                                                           mu_s[i_source, 1] *
                                                           self.arrow_mult,
                                                           **self.arrow_kwargs)))

        # Turn off all ticks
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.get_xaxis().set_ticklabels([])
        self.ax.get_yaxis().set_ticklabels([])

        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)

        # Animate if required, otherwise return still image
        if animate:
            anim = FuncAnimation(
                fig, self.update, interval=interval, frames=n_frames, fargs=[dt])
            return anim
        else:
            return fig

    def update(self, frame_number, dt):
        """ Update animation """

        mu_s = np.zeros((self.n_total, 2))
        theta_s = np.zeros((self.n_total, 2))

        for i_lens in range(self.n_lens):
            b_ary = np.transpose([self.sources["theta_x"] - self.lenses["theta_x"][i_lens],
                                  self.sources["theta_y"] - self.lenses["theta_y"][i_lens]]) * asctorad

            vel_l = np.array([self.lenses["v_x"][i_lens],
                              self.lenses["v_y"][i_lens]])
            for i_source in range(self.n_total):
                mu_s[i_source] += self.mu(b_ary[i_source], vel_l / self.lenses["D"][i_lens],
                                          self.lenses["R_0"][i_lens],
                                          self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])
                theta_s[i_source] += self.theta(b_ary[i_source], self.lenses["R_0"][i_lens],
                                                self.lenses["M_0"][i_lens], self.lenses["D"][i_lens])

            mu_l = (vel_l / self.lenses["D"][i_lens]) / (Year ** -1) * radtoasc

            self.lenses["theta_x"][i_lens] = self.lenses["theta_x"][i_lens] + mu_l[0] * dt
            self.lenses["theta_y"][i_lens] = self.lenses["theta_y"][i_lens] + mu_l[1] * dt

        self.sources["theta_x"] = self.sources["theta_x_0"] + theta_s[:, 0]
        self.sources["theta_y"] = self.sources["theta_y_0"] + theta_s[:, 1]

        self.scatter.set_offsets(np.transpose(
            [self.sources["theta_x"], self.sources["theta_y"]]))

        if self.show_lens:
            im = np.zeros_like(self.x_grid)

            for i_lens in range(self.n_lens):
                self.x_grid, self.y_grid = np.meshgrid(self.x_coords - self.lenses["theta_x"][i_lens],
                                                       self.y_coords - self.lenses["theta_y"][i_lens])
                r_grid = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
                im += \
                    Profiles.MdMdb_Gauss(r_grid, self.lenses["R_0"][i_lens] / self.lenses["D"][i_lens] * radtoasc,
                                         self.lenses["M_0"][i_lens])[0]

                self.imshow.set_array(im)

        if self.show_vel_arrow:
            for i_source in range(len(mu_s)):
                self.arrows[i_source].remove()
                self.arrows[i_source] = self.ax.add_patch(Arrow(self.sources["theta_x"][i_source],
                                                                self.sources["theta_y"][i_source],
                                                                mu_s[i_source, 0] *
                                                                self.arrow_mult,
                                                                mu_s[i_source, 1] *
                                                                self.arrow_mult,
                                                                **self.arrow_kwargs))

    @classmethod
    def mu(self, beta_vec, v_ang_vec, R_0, M_0, d_lens):
        """ Get lens-induced proper motion vector
        """

        # Convert angular to physical impact parameter
        b_vec = d_lens * np.array(beta_vec)
        # Convert angular to physical velocity
        v_vec = d_lens * np.array(v_ang_vec)
        b = np.linalg.norm(b_vec)  # Impact parameter
        M, dMdb, _ = Profiles.MdMdb_Gauss(b, R_0, M_0)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter
        b_dot_v = np.dot(b_unit_vec, v_vec)
        factor = (dMdb / b * b_unit_vec * b_dot_v
                  + M / b ** 2 * (v_vec - 2 * b_unit_vec * b_dot_v))

        return -factor * 4 * GN / (asctorad / Year)  # Convert to as/yr

    @classmethod
    def theta(self, beta_vec, R_0, M_0, d_lens):
        """ Get lens-induced deflection vector
        """

        # Convert angular to physical impact parameter
        b_vec = d_lens * np.array(beta_vec)
        b = np.linalg.norm(b_vec)  # Impact parameter
        M, _, _ = Profiles.MdMdb_Gauss(b, R_0, M_0)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter

        return 4 * GN * M / b * b_unit_vec * radtoasc  # Convert to as
