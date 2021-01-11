import numpy as np
import mpmath as mp
from scipy.special import kn
from tqdm import *

from theory.units import *
from theory.profiles import Profiles


class PowerSpectra(Profiles):
    def __init__(self, precompute=["Burk", "NFW"]):
        """ Class to calculate expected power spectra from astrometric induced velocities and accelerations

            :param precompute: List of profiles to precompute arrays for to speed up computation ['Burk', 'NFW']
        """

        Profiles.__init__(self)

        # Precompute arrays to speed up computation
        if "Burk" in precompute:
            self.precompute_MBurkdivM0()
        if "NFW" in precompute:
            self.precompute_MNFWdivM0()

    ##################################################
    # Induced velocity/acceleration power spectra
    ##################################################

    def Cl_Gauss(self, R0, M0, Dl, v, l):
        """ Induced velocity power spectrum for Gaussian lens
            :param R0: size of lens
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        beta0 = R0 / Dl
        return (4 * GN * M0 * v / Dl ** 2) ** 2 * np.pi / 2.0 * np.exp(-(l ** 2) * beta0 ** 2)

    def Cl_Plummer(self, R0, M0, Dl, v, l):
        """ Induced velocity power spectrum for Plummer lens
            :param R0: size of lensfs
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        beta0 = R0 / Dl
        return (4 * GN * M0 * v / Dl ** 2) ** 2 * np.pi / 2.0 * l ** 2 * beta0 ** 2 * kn(1, l * beta0) ** 2

    def Cl_Point(self, M0, Dl, v, l):
        """ Induced velocity power spectrum for point lens
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        return (4 * GN * M0 * v / Dl ** 2) ** 2 * np.pi / 2.0

    def Cl_NFW(self, M200, Dl, v, l, Rsub=None):
        """ Induced velocity power spectrum for NFW lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        if self.c200_model is self.c200_Moline:
            kwargs = {"xsub": Rsub / R200_MW}
        else:
            kwargs = {}

        r_s, rho_s = self.get_rs_rhos_NFW(M200, **kwargs)

        M0 = 4 * np.pi * r_s ** 3 * rho_s
        pref = GN ** 2 * v ** 2 * 8 * np.pi * l ** 2 / Dl ** 4
        theta_s = r_s / Dl
        if not self.precompute_NFW:
            MjdivM0 = self.MNFWdivM0_integ(theta_s, l)
        else:
            MjdivM0 = 10 ** self.MNFWdivM0_integ_interp(np.log10(l), np.log10(theta_s))[0]
        return pref * M0 ** 2 * MjdivM0 ** 2

    def Cl_tNFW(self, M200, Dl, v, l, tau=15.0):
        """ Induced velocity power spectrum for truncated NFW lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
            :param tau: ratio of truncation and scale radii
        """
        r_s, rho_s = self.get_rs_rhos_NFW(M200)
        M0 = 4 * np.pi * r_s ** 3 * rho_s
        pref = GN ** 2 * v ** 2 * 8 * np.pi * l ** 2 / Dl ** 4
        theta_s = r_s / Dl
        self.set_mp()
        MjdivM0 = mp.quadosc(lambda theta: self.MtNFWdivM0(theta / theta_s, tau) * mp.j1(l * theta), [0, mp.inf], period=2 * mp.pi / l)
        return pref * M0 ** 2 * MjdivM0 ** 2

    def Cl_Burk(self, M200, Dl, v, l, p=0.7):
        """ Induced velocity power spectrum for Burkert lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
            :param p: ratio of NFW and Burkert concentrations
        """
        r_b, rho_b = self.get_rb_rhob_Burk(M200, p)
        M0 = 4 * np.pi * r_b ** 3 * rho_b
        pref = GN ** 2 * v ** 2 * 8 * np.pi * l ** 2 / Dl ** 4
        theta_b = r_b / Dl
        if not self.precompute_Burk:
            MjdivM0 = self.MBurkdivM0_integ(theta_b, l)
        else:
            MjdivM0 = 10 ** self.MBurkdivM0_integ_interp(np.log10(l), np.log10(theta_b))[0]
        return pref * M0 ** 2 * MjdivM0 ** 2


class PowerSpectraPopulations(PowerSpectra):
    def __init__(self, l_min=1, l_max=5000, n_l=50):
        """
        Class to calculate power spectra of populations

        :param l_min: Minimum multipole
        :param l_max: Maximum multipole
        :param n_l: Number of multipoles interpolated over
        """

        PowerSpectra.__init__(self)

        self.l_min = l_min
        self.l_max = l_max
        self.n_l = n_l

        self.l_ary = np.arange(self.l_min, self.l_max)
        self.l_ary_calc = np.logspace(np.log10(self.l_min), np.log10(self.l_max), self.n_l)

        self.calc_v_proj_mean_integrals()

    def integrand_norm(self, x):
        """ Integrand for calculating overall normalization of 
            joint mass-radial distribution pdf
        """
        M200, r = np.exp(x[0]) * M_s, np.exp(x[1]) * kpc
        return 4 * np.pi * M200 * r * self.rho_M(M200, **self.rho_M_kwargs) * self.rho_R(r)

    def integrand_norm_mass(self, x):
        """ Integrand for calculating overall normalization of
            joint mass-radial distribution pdf
        """
        M200 = np.exp(x[0]) * M_s
        return M200 * self.rho_M(M200, **self.rho_M_kwargs)

    def integrand_norm_compact(self, x):
        """ Integrand for calculating overall normalization of 
            joint mass-radial distribution pdf for compact objects
        """
        r = np.exp(x[0]) * kpc
        return 4 * np.pi * r * self.rho_R(r)

    def set_radial_distribution(self, rho_R, R_min, R_max, **kwargs):
        """
        Set radial distribution of population

        :param rho_R: Galactocentric number density
        :param R_min: Minimum Galactocentric radius
        :param R_max: Maximum Galactocentric radius
        :param kwargs: keyword arguments to pass to rho_R function
        :return:
        """

        self.rho_R = rho_R
        self.rho_R_kwargs = kwargs

        self.R_min = R_min
        self.R_max = R_max

    def set_mass_distribution(self, rho_M, M_min, M_max, M_min_calib, M_max_calib, N_calib, **kwargs):
        #
        """
        Set mass distribution of population
        TODO: Stabilize distributions

        :param rho_M: Mass function
        :param M_min: Minimum subhalo mass
        :param M_max: Maximum subhalo mass
        :param M_min_calib: Minimum subhalo mass for calibration
        :param M_max_calib: Maximum subhalo mass for calibration
        :param N_calib: Number of subhalos between [M_min_calib, M_max_calib]
        :param kwargs: keyword arguments to pass to rho_M function
        :return:
        """

        self.rho_M = rho_M
        self.rho_M_kwargs = kwargs

        self.M_min = M_min
        self.M_max = M_max

        self.M_min_calib = M_min_calib
        self.M_max_calib = M_max_calib

        self.N_calib = N_calib

        logM_integ_ary = np.linspace(np.log(self.M_min_calib / M_s), np.log(self.M_max_calib / M_s), 500)
        logR_integ_ary = np.linspace(np.log(self.R_min / kpc), np.log(self.R_max / kpc), 500)

        measure = (logM_integ_ary[1] - logM_integ_ary[0]) * (logR_integ_ary[1] - logR_integ_ary[0])

        norm = 0

        for logM in logM_integ_ary:
            for logR in logR_integ_ary:
                norm += self.integrand_norm([logM, logR])

        norm *= measure

        logM_integ_ary = np.linspace(np.log(self.M_min / M_s), np.log(self.M_max / M_s), 500)
        measure_mass = logM_integ_ary[1] - logM_integ_ary[0]

        self.norm_mass = 0

        for logM in logM_integ_ary:
            self.norm_mass += self.integrand_norm_mass([logM])

        self.norm_mass *= measure_mass

        self.pref = self.N_calib / norm

        l_los_ary = np.logspace(-5, 5, 200) * pc
        n_lens_ary = [self.n_lens(l_los_max=l_max) for l_max in l_los_ary]
        self.l_cutoff = 10 ** np.interp(0, np.log10(n_lens_ary), np.log10(l_los_ary))

    def set_mass_distribution_compact(self, M_DM, f_DM, R0_DM=0):
        # TODO: Stabilize distributions

        self.M_DM = M_DM
        self.R0_DM = R0_DM
        self.f_DM = f_DM

        logR_integ_ary = np.linspace(np.log(self.R_min / kpc), np.log(self.R_max / kpc), 500)

        measure = logR_integ_ary[1] - logR_integ_ary[0]

        norm = 0

        for logR in logR_integ_ary:
            norm += self.integrand_norm_compact([logR])

        norm *= measure

        self.pref = self.f_DM * 1.0 / (self.M_DM / (1e12 * M_s)) / norm

        l_los_ary = np.logspace(-2, 2, 100) * kpc
        n_lens_ary = [self.n_lens_compact(l_los_max=l_max) for l_max in l_los_ary]
        self.l_cutoff = 10 ** np.interp(0, np.log10(n_lens_ary), np.log10(l_los_ary))

    def n_lens(self, l_los_max, l_los_min=0 * kpc):
        """
        Number of lenses between los distance l_los_min and l_los_max:
        """

        def integrand_n_lens(x):
            l, theta = x[0], x[1]
            r = np.sqrt(l ** 2 + Rsun ** 2 - 2 * l * Rsun * np.cos(theta))
            return self.rho_R(r, **self.rho_R_kwargs) / r ** 2 * l ** 2

        l_los_integ_ary = np.linspace(l_los_min, l_los_max, 50)
        # logM_integ_ary = np.linspace(np.log(self.M_min / M_s), np.log(self.M_max / M_s), 50)
        theta_integ_ary = np.linspace(0, np.pi, 50)

        measure = (l_los_integ_ary[1] - l_los_integ_ary[0]) * (theta_integ_ary[1] - theta_integ_ary[0])

        integ = 0

        for l_los in l_los_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta) * integrand_n_lens([l_los, theta])
        integ *= 2 * np.pi * measure

        return self.pref * integ * self.norm_mass

    def n_lens_compact(self, l_los_max, l_los_min=0 * kpc):
        """
        Number of lenses between los distance l_los_min and l_los_max:
        """

        def integrand_n_lens(x):
            l, theta = x[0], x[1]
            r = np.sqrt(l ** 2 + Rsun ** 2 - 2 * l * Rsun * np.cos(theta))
            return self.rho_R(r, **self.rho_R_kwargs) / r ** 2 * l ** 2

        l_los_integ_ary = np.linspace(l_los_min, l_los_max, 50)
        theta_integ_ary = np.linspace(0, np.pi, 50)

        measure = (l_los_integ_ary[1] - l_los_integ_ary[0]) * (theta_integ_ary[1] - theta_integ_ary[0])

        integ = 0

        for l_los in l_los_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta) * integrand_n_lens([l_los, theta])
        integ *= 2 * np.pi * measure

        return self.pref * integ

    def set_subhalo_properties(self, c200_model):
        """
        Set properties of subhalo profile (just concentration-mass relation for now)

        :param c200_model: Concentration-mass relation
        """
        self.c200_model = c200_model

    def calc_v_proj_mean_integrals(self):

        # Mean projected v**2 for velocity integral
        self.vsq_proj_mean = 5.789779157814031e-07

        # Mean projected v**4 for acceleration integral
        self.v4_proj_mean = 5.914900709341262e-13

    def integrand_los(self, x, ell, accel=False):
        """
        Population integrand in Earth frame
        """
        logl, theta, logm = x[0], x[1], x[2]
        m = np.exp(logm) * M_s
        l = np.exp(logl) * kpc

        r = np.sqrt(l ** 2 + Rsun ** 2 - 2 * l * Rsun * np.cos(theta))

        if accel:
            pref = (3 / 4) * ell ** 2 / l ** 2
            units = (1e-6 * asctorad / Year ** 2) ** 2
        else:
            pref = 1
            units = (1e-6 * asctorad / Year) ** 2

        return pref * l * m * self.Cl_NFW(m, l, 1, ell, r) / units * self.rho_M(m, **self.rho_M_kwargs) * self.rho_R(r, **self.rho_R_kwargs) * l ** 2 / r ** 2

    def integrand_gc(self, x, ell, accel=False):
        """
        Population integrand in Galactocentric frame
        """
        logR, theta, logm = x[0], x[1], x[2]
        m = np.exp(logm) * M_s
        r = np.exp(logR) * kpc

        if accel:
            pref = (3 / 4) * ell ** 2 / r ** 2
            units = (1e-6 * asctorad / Year ** 2) ** 2
        else:
            pref = 1
            units = (1e-6 * asctorad / Year) ** 2

        l = np.sqrt(r ** 2 + Rsun ** 2 + 2 * r * Rsun * np.cos(theta))
        return pref * r * m * self.Cl_NFW(m, l, 1, ell, r) / units * self.rho_M(m, **self.rho_M_kwargs) * self.rho_R(r, **self.rho_R_kwargs)

    def integrand_compact(self, x, ell, accel=False):
        """
        Population integrand for compact objects
        """
        logl, theta = x[0], x[1]
        l = np.exp(logl) * kpc

        r = np.sqrt(l ** 2 + Rsun ** 2 - 2 * l * Rsun * np.cos(theta))

        if accel:
            pref = (3 / 4) * ell ** 2 / l ** 2
            units = (1e-6 * asctorad / Year ** 2) ** 2
        else:
            pref = 1
            units = (1e-6 * asctorad / Year) ** 2

        if not self.R0_DM == 0:
            return pref * l * self.Cl_Gauss(self.R0_DM, self.M_DM, l, 1, ell) / units * self.rho_R(r, **self.rho_R_kwargs) * l ** 2 / r ** 2
        else:
            return pref * l * self.Cl_Point(self.M_DM, l, 1, ell) / units * self.rho_R(r, **self.rho_R_kwargs) * l ** 2 / r ** 2

    def C_l_total(self, ell, theta_deg_mask=0, l_los_min=0.1 * kpc, l_los_max=200 * kpc, accel=False):
        """
        Get total population power spectrum at given multipole
        TODO: double check number of integration points and integration ranges

        :param ell: Multipole
        :param theta_deg_mask: Whether to apply a radial angular mask
        :param l_min: Smallest distance to integrate from
        :param l_max: Largest distance to integrate to
        :param accel: Whether this is the acceleration PS (otherwise velocity)
        :return: Power spectrum at given multipole ell
        """
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logl_integ_ary = np.linspace(np.log(l_los_min / kpc), np.log(l_los_max / kpc), 50)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi - theta_rad_mask, 20)
        logM_integ_ary = np.linspace(np.log(self.M_min / M_s), np.log(self.M_max / M_s), 20)

        measure = (logl_integ_ary[1] - logl_integ_ary[0]) * (theta_integ_ary[1] - theta_integ_ary[0]) * (logM_integ_ary[1] - logM_integ_ary[0])

        integ = 0
        for logl in logl_integ_ary:
            for theta in theta_integ_ary:
                for logM in logM_integ_ary:
                    integ += np.sin(theta) * self.integrand_los([logl, theta, logM], ell, accel)
        integ *= 2 * np.pi * measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def C_l_compact_total(self, ell, theta_deg_mask=0, l_los_min=0.1 * kpc, l_los_max=200 * kpc, accel=False):
        """
        Get total population power spectrum at given multipole for compact objects
        TODO: double check number of integration points and integration ranges
        :param ell: Multipole
        :param theta_deg_mask: Whether to apply a radial angular mask
        :param accel: Whether this is the acceleration PS (otherwise velocity)
        :return: Power spectrum at given multipole ell
        """

        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logl_integ_ary = np.linspace(np.log(l_los_min / kpc), np.log(l_los_max / kpc), 200)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi - theta_rad_mask, 50)

        measure = (logl_integ_ary[1] - logl_integ_ary[0]) * (theta_integ_ary[1] - theta_integ_ary[0])

        integ = 0

        for logl in logl_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta) * self.integrand_compact([logl, theta], ell, accel)
        integ *= 2 * np.pi * measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def dC_l_dR_total(self, ell, R, theta_deg_mask=0, accel=False):
        """
        Differential PS at given radius R
        """
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        theta_integ_ary = np.linspace(theta_rad_mask, np.pi - theta_rad_mask, 500)
        logM_integ_ary = np.linspace(np.log(self.M_min / M_s), np.log(self.M_max / M_s), 20)

        measure = (theta_integ_ary[1] - theta_integ_ary[0]) * (logM_integ_ary[1] - logM_integ_ary[0])

        integ = 0

        for theta in theta_integ_ary:
            for logM in logM_integ_ary:
                integ += np.sin(theta) * self.integrand_gc([np.log(R / kpc), theta, logM], ell, accel) / R
        integ *= 2 * np.pi * measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def dC_l_dM_total(self, ell, M, theta_deg_mask=0, accel=False):
        """
        Differential PS at given mass M
        """
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logR_integ_ary = np.linspace(np.log(self.R_min / kpc), np.log(self.R_max / kpc), 100)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi - theta_rad_mask, 100)

        measure = (logR_integ_ary[1] - logR_integ_ary[0]) * (theta_integ_ary[1] - theta_integ_ary[0])

        integ = 0

        for logR in logR_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta) * self.integrand_gc([logR, theta, np.log(M / M_s)], ell, accel) / M
        integ *= 2 * np.pi * measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def get_C_l_total_ary(self, theta_deg_mask=0, accel=False, l_los_min=5 * pc, l_los_max=200 * kpc):
        """
        Get power spectrum over full multipole range
        """
        self.C_l_calc_ary = [self.C_l_total(ell, theta_deg_mask=theta_deg_mask, accel=accel, l_los_min=l_los_min, l_los_max=l_los_max) for ell in tqdm_notebook(self.l_ary_calc)]
        self.C_l_ary = 10 ** np.interp(np.log10(self.l_ary), np.log10(self.l_ary_calc), np.log10(self.C_l_calc_ary))
        return self.C_l_ary

    def get_C_l_compact_total_ary(self, theta_deg_mask=0, accel=False, l_los_min=1e-3 * pc, l_los_max=10 * kpc):
        """
        Get power spectrum over full multipole range for compact objects
        """

        if self.R0_DM == 0:
            if accel:
                self.C_l_ary = self.l_ary ** 2 * np.array(len(self.l_ary) * [self.C_l_compact_total(1, theta_deg_mask=theta_deg_mask, accel=True, l_los_min=l_los_min, l_los_max=l_los_max)])
            else:
                self.C_l_ary = len(self.l_ary) * [self.C_l_compact_total(1, theta_deg_mask=theta_deg_mask, accel=False, l_los_min=l_los_min, l_los_max=l_los_max)]
        else:
            self.C_l_calc_ary = [self.C_l_compact_total(ell, theta_deg_mask=theta_deg_mask, accel=accel, l_los_min=l_los_min, l_los_max=l_los_max) for ell in (self.l_ary_calc)]
            self.C_l_ary = 10 ** np.interp(np.log10(self.l_ary), np.log10(self.l_ary_calc), np.log10(self.C_l_calc_ary))
        return np.array(self.C_l_ary)
