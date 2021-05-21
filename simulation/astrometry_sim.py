import numpy as np
import pandas as pd
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import *

from theory.units import *
from simulation.subhalo_sim import SubhaloSample
from simulation.estimator_wholesky import get_vector_alm, get_cross_correlation_Cells
from theory.profiles import Profiles


class QuasarSim(SubhaloSample):
    def __init__(self, verbose=True, max_sep=3, data_dir='../data/', save_tag='sample', save=False, save_dir='Output/',
                 sim_uniform=False, nside=512, calc_powerspecs=True, do_alpha=False, *args, **kwargs):
        """ Class for simulating lens-induced astrometric perturbations
            in Gaia DR2 QSOs or a uniform quasar sample. 

            :param verbose: whether to print out progress
            :param max_sep: maximum seperation in degrees to consider between lenses and quasars
            :param data_dir: local folder with external data
            :param save_tag: tag to save files under
            :param save: whether to save output
            :param save_dir: directory to output simulation results to
            :param sim_uniform: whether to simulate a uniform sample for signal studies
            :param nside: nside of uniform sample
            :param calc_powerspecs: whether to calculate vector power spectra
            :param do_alpha: whether to calculate induced accelerations
            *args, **kwargs: lens population parameters passed to the SubhaloSample class 
        """
        SubhaloSample.__init__(self, *args, **kwargs)

        self.max_sep = max_sep
        self.verbose = verbose
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.save_tag = save_tag
        self.save = save
        self.calc_powerspecs = calc_powerspecs
        self.do_alpha = do_alpha

        if self.verbose:
            self.tqdm_function = tqdm_notebook
        else:
            self.tqdm_function = tqdm

        # Either simulate a uniform sample of background sources
        # or use Gaia DR2 quasars are source locations
        if sim_uniform:
            self.load_uniform_sample(nside)
        else:
            self.load_gaia_quasars()

        # self.analysis_pipeline()

    def analysis_pipeline(self, get_sample=True):
        """ Run analysis sequence
        """
        if get_sample:
            self.get_sh_sample()
        self.get_mu_alpha()
        self.get_powerspecs()
        if self.save:
            self.save_products()

    def update_sample(self, *args, **kwargs):
        """ Update subhalo population parameters and redo simulation

            *args, **kwargs: lens population parameters passed to the SubhaloSample class
        """
        SubhaloSample.__init__(self, *args, **kwargs)
        self.analysis_pipeline()

    def load_gaia_quasars(self):
        """ Load quasars and their positions
        """
        pd_qsrs = pd.read_csv(self.data_dir + "quasars_phot.csv")

        ra = pd_qsrs['ra'].values
        dec = pd_qsrs['dec'].values

        # coords_qsrs_icrs = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        # self.coords_qsrs = coords_qsrs_icrs.transform_to('galactic')

        self.coords_qsrs = SkyCoord(
            l=ra * u.deg, b=dec * u.deg, frame='galactic')
        self.n_qsrs = len(pd_qsrs)

    def load_uniform_sample(self, nside):
        """ Load a uniform sample of background sources
            located on the centers of Healpix pixels
        """
        l_ary, b_ary = hp.pix2ang(nside, np.arange(
            hp.nside2npix(nside)), lonlat=True)
        self.coords_qsrs = SkyCoord(
            l=l_ary * u.deg, b=b_ary * u.deg, frame='galactic')
        self.n_qsrs = hp.nside2npix(nside)

    def get_angular_sep_dir(self, pos1, pos2):
        """ Get unit separation vector in (l, b) direction
            pointing from pos2 to pos1 in the plane of pos2
        """

        # pos1 is quasar/source,  pos2 is lens
        l1, b1 = pos1
        l2, b2 = pos2

        theta_lens, phi_lens = np.radians(90 - b2), np.radians(l2)
        theta_src, phi_src = np.radians(90 - b1), np.radians(l1)

        # Get unit vectors at position of lens
        theta_hat = [np.cos(theta_lens) * np.cos(phi_lens),
                     np.cos(theta_lens) * np.sin(phi_lens), -np.sin(theta_lens)]
        phi_hat = [-np.sin(phi_lens), np.cos(phi_lens), 0]

        # Get vectors corresponding to pos1 and pos2
        vec1 = np.array([np.sin(theta_src) * np.cos(phi_src),
                         np.sin(theta_src) * np.sin(phi_src), np.cos(theta_src)])
        vec2 = np.array(
            [np.sin(theta_lens) * np.cos(phi_lens), np.sin(theta_lens) * np.sin(phi_lens), np.cos(theta_lens)])

        dvec = vec1 - vec2

        proj_dvec = dvec - (np.dot(vec1, vec2) * vec2 - vec2)
        proj_dvec = proj_dvec / np.linalg.norm(proj_dvec)

        # Get components of displacement vector along theta and phi
        # Reverse theta direction since regular polar angle points down from N
        return np.array([np.dot(proj_dvec, phi_hat), -np.dot(proj_dvec, theta_hat)])

    def get_angular_sep_dir_vec(self, pos1, pos2):
        """ Get unit separation vector in (l, b) direction
            pointing from pos2 to pos1 in the plane of pos2
        """

        # pos1 is quasar/source,  pos2 is lens
        l1_vec, b1_vec = pos1
        l2, b2 = pos2

        theta_lens, phi_lens = np.radians(90 - b2), np.radians(l2)
        theta_src_vec, phi_src_vec = np.radians(90 - b1_vec), np.radians(l1_vec)

        # Get unit vectors at position of lens
        theta_hat = [np.cos(theta_lens) * np.cos(phi_lens),
                     np.cos(theta_lens) * np.sin(phi_lens), -np.sin(theta_lens)]
        phi_hat = [-np.sin(phi_lens), np.cos(phi_lens), 0]

        theta_hat = np.reshape(theta_hat, (3, 1))
        phi_hat = np.reshape(phi_hat, (3, 1))

        # Get vectors corresponding to pos1 and pos2
        vec1 = np.array([np.sin(theta_src_vec) * np.cos(phi_src_vec),
                         np.sin(theta_src_vec) * np.sin(phi_src_vec), np.cos(theta_src_vec)])
        vec2 = np.array(
            [np.sin(theta_lens) * np.cos(phi_lens), np.sin(theta_lens) * np.sin(phi_lens), np.cos(theta_lens)])

        vec2 = np.reshape(vec2, (3, 1))
        dvec = vec1 - vec2
        
        proj_dvec = dvec - (np.einsum('ij,ik->kj', vec1, vec2) * vec2 - vec2)
        proj_dvec = proj_dvec / np.linalg.norm(proj_dvec, axis=0)

        # Get components of displacement vector along theta and phi
        # Reverse theta direction since regular polar angle points down from N
        return np.array([np.einsum('ij,ik->kj', proj_dvec, phi_hat)[0], -np.einsum('ij,ik->kj', proj_dvec, theta_hat)[0]])


    def get_mu_alpha(self):
        """ Get induced velocities (and optionally accelerations) of quasars and store them in self.mu_qsrs
        """

        # Get indices and angular separations of objects within `max_sep` of the lenses with indices `idxs_lenses`
        idxs_lenses, idxs_qsrs, self.sep2d, _ = self.coords_qsrs.search_around_sky(self.coords_galactic,
                                                                                   self.max_sep * u.deg)
        self.sep2d = np.deg2rad(self.sep2d.value)

        # Loop over lenses and get induced velocities (and optionally accelerations)
        # for nearby (within `max_sep`) quasars

        self.mu_qsrs = np.zeros((self.n_qsrs, 2))
        self.alpha_qsrs = np.zeros((self.n_qsrs, 2))

        for i_lens in self.tqdm_function(range(self.N_halos), disable=True):

            # Lens velocity
            self.pm_l_cosb_lens = self.coords_galactic.pm_l_cosb.value[i_lens]
            self.pm_b_lens = self.coords_galactic.pm_b.value[i_lens]

            # Convert lens velocity from mas/yr to natural units
            v_lens = np.array(
                [self.pm_l_cosb_lens, self.pm_b_lens]) * 1e-3 * asctorad / Year

            # Indices of quasars around lens i_lens
            idxs_qsrs_around = idxs_qsrs[idxs_lenses == i_lens]

            # Angular separation between sources and lens i_lens
            beta_ary = np.array(self.sep2d[idxs_lenses == i_lens])

            # Lens position
            self.l_lens = self.coords_galactic.l.value[i_lens]
            self.b_lens = self.coords_galactic.b.value[i_lens]

            beta_ary = np.reshape(beta_ary, (1, len(beta_ary)))

            # Angular impact parameters of quasars around lens i_lens
            self.beta_lens_qsrs_around = beta_ary * self.get_angular_sep_dir_vec(
                [self.coords_qsrs.l.value[idxs_qsrs_around], self.coords_qsrs.b.value[idxs_qsrs_around]],
                [self.l_lens, self.b_lens])

            self.beta_lens_qsrs_around = np.transpose(self.beta_lens_qsrs_around)

            # Grab some lens properties
            c200_lens, m_lens, d_lens = self.c200_sample[i_lens], self.M_sample[i_lens], self.d_sample[i_lens]

            # Populate quasar induced velocity (and optionally accelerations) array
            self.mu_qsrs[idxs_qsrs_around] += self.mu_vec(d_lens * self.beta_lens_qsrs_around, d_lens * v_lens, c200_lens, m_lens)

            # Populate accelerations. Not implemented for vectorized version of code yet.
            if self.do_alpha:
                self.alpha_qsrs[idxs_qsrs_around] += self.alpha_vec(d_lens * self.beta_lens_qsrs_around, d_lens * v_lens, c200_lens, m_lens)

    def mu(self, b_vec, v_vec, *args_lens):
        """ Get lens-induced velocity

            :param b_vec: physical impact parameter (pointing lens to source) vector in rad
            :param v_vec: physical velocity in natural units
            :param args_lens: arguments going into enclosed mass function
            :return: lens-induced velocity in as/yr
        """

        b = np.linalg.norm(b_vec)  # Impact parameter

        if self.sh_profile == "NFW":
            MdMdb_func = self.MdMdb_NFW
        elif self.sh_profile == "Plummer":
            MdMdb_func = self.MdMdb_Plummer
        elif self.sh_profile == "Gaussian":
            MdMdb_func = self.MdMdb_Gauss
        else:
            raise Exception("Unknown profile specification!")

        M, dMdb, _ = MdMdb_func(b, *args_lens)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter
        b_dot_v = np.dot(b_unit_vec, v_vec)
        factor = (dMdb / b * b_unit_vec * b_dot_v
                  + M / b ** 2 * (v_vec - 2 * b_unit_vec * b_dot_v))

        return -factor * 4 * GN / (asctorad / Year)  # Convert to as/yr

    def mu_vec(self, b_vec, v_vec, *args_lens):
        """ Get lens-induced velocity

            :param b_vec: physical impact parameter (pointing lens to source) vector in rad
            :param v_vec: physical velocity in natural units
            :param args_lens: arguments going into enclosed mass function
            :return: lens-induced velocity in as/yr
        """

        b = np.linalg.norm(b_vec, axis=1)  # Impact parameter

        if self.sh_profile == "NFW":
            MdMdb_func = self.MdMdb_NFW
        elif self.sh_profile == "Plummer":
            MdMdb_func = self.MdMdb_Plummer
        elif self.sh_profile == "Gaussian":
            MdMdb_func = self.MdMdb_Gauss
        else:
            raise Exception("Unknown profile specification!")

        M, dMdb, _ = MdMdb_func(b, *args_lens)

        b_unit_vec = np.transpose(b_vec) / b  # Convert angular to physical impact parameter
        b_unit_vec = np.transpose(b_unit_vec)

        b_dot_v = np.dot(b_unit_vec, v_vec)
        b_dot_v = np.transpose(b_dot_v)
        b_unit_vec = np.transpose(b_unit_vec)
        v_vec = np.reshape(v_vec, (2, 1))
        # print(np.shape(b_dot_v), np.shape(b_unit_vec))

        factor = (dMdb / b * b_unit_vec * b_dot_v
                  + M / b ** 2 * (v_vec - 2 * b_unit_vec * b_dot_v))

        return np.transpose(-factor * 4 * GN / (asctorad / Year))  # Convert to as/yr

    def alpha(self, b_vec, v_vec, *args_lens):
        """ Get lens-induced acceleration

            :param b_vec: physical impact parameter (pointing lens to source) vector in rad
            :param v_vec: physical velocity in natural units
            :param args_lens: arguments going into enclosed mass function
            :return: lens-induced acceleration in as/yr^2
        """
        b = np.linalg.norm(b_vec)  # Impact parameter

        if self.sh_profile == "NFW":
            MdMdb_func = self.MdMdb_NFW
        elif self.sh_profile == "Plummer":
            MdMdb_func = self.MdMdb_Plummer
        elif self.sh_profile == "Gaussian":
            MdMdb_func = self.MdMdb_Gauss
        else:
            raise Exception("Unknown profile specification!")

        M, dMdb, d2Mdb2 = MdMdb_func(b, *args_lens)
        b_unit_vec = b_vec / b  # Convert angular to physical impact parameter
        b_dot_v = np.dot(b_unit_vec, v_vec)
        factor = (6 * b_vec * b_dot_v ** 2 * M - 2 * b *
                  ((2 * v_vec * b_dot_v + b_vec * np.dot(v_vec,
                                                         (v_vec * b - b_vec * b_dot_v) / b ** 2)) * M +
                   2 * b_vec * b_dot_v ** 2 * dMdb) + b ** 2 *
                  ((2 * v_vec * b_dot_v + b_vec * np.dot(v_vec, (v_vec * b - b_vec * b_dot_v) / b ** 2))
                   * dMdb + b_vec * b_dot_v ** 2 * d2Mdb2)) / b ** 4

        return -factor * 4 * GN / (asctorad / Year ** 2)  # Convert to as/yr

    def alpha_vec(self, b_vec, v_vec, *args_lens):
        """ Get lens-induced acceleration

            :param b_vec: physical impact parameter (pointing lens to source) vector in rad
            :param v_vec: physical velocity in natural units
            :param args_lens: arguments going into enclosed mass function
            :return: lens-induced acceleration in as/yr^2
        """
        raise NotImplementedError
        
    def get_powerspecs(self):
        """ Calculate vector spherical harmonic coefficients
        """
        if self.calc_powerspecs:
            self.Cl_B, self.Cl_C, self.fB, self.fC = get_vector_alm(
                self.mu_qsrs[:, 1], self.mu_qsrs[:, 0])
            if self.do_alpha:
                self.Cl_B_alpha, self.Cl_C_alpha, self.fB_alpha, self.fC_alpha = get_vector_alm(self.alpha_qsrs[:, 1],
                                                                                                self.alpha_qsrs[:, 0])
                self.Cl_B_mu_alpha = get_cross_correlation_Cells(
                    self.fB, self.fB_alpha)
                self.Cl_C_mu_alpha = get_cross_correlation_Cells(
                    self.fC, self.fC_alpha)
            else:
                self.Cl_B_alpha, self.Cl_C_alpha, self.fB_alpha, self.fC_alpha = [], [], [], []
                self.Cl_B_mu_alpha, self.Cl_C_mu_alpha = [], []

        else:
            self.Cl_B, self.Cl_C, self.fB, self.fC = [], [], [], []
            self.Cl_B_alpha, self.Cl_C_alpha, self.fB_alpha, self.fC_alpha = [], [], [], []
            self.Cl_B_mu_alpha, self.Cl_C_mu_alpha = [], []

    def save_products(self):
        """ Save sim output
        """
        np.savez(self.save_dir + '/' + self.save_tag,
                 pos_lens=[self.coords_galactic.l.value,
                           self.coords_galactic.b.value],
                 vel_lens=[self.coords_galactic.pm_l_cosb.value,
                           self.coords_galactic.pm_b.value],
                 mu_qsrs=np.transpose(self.mu_qsrs),  # in as/yr
                 Cl_B=self.Cl_B,
                 Cl_C=self.Cl_C,
                 fB=self.fB,
                 fC=self.fC,
                 alpha_qsrs=np.transpose(self.alpha_qsrs),  # in as/yr^2
                 Cl_B_alpha=self.Cl_B_alpha,
                 Cl_C_alpha=self.Cl_C_alpha,
                 fB_alpha=self.fB_alpha,
                 fC_alpha=self.fC_alpha,
                 Cl_B_mu_alpha=self.Cl_B_mu_alpha,
                 Cl_C_mu_alpha=self.Cl_C_mu_alpha
                 )
