import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from astropy.cosmology import Planck15
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.optimize import minimize, fsolve
from scipy.special import erfc
from classy import Class
import random
import string

from theory.spec_calc import PowerSpectra, PowerSpectraPopulations
from theory.astrometry_forecast import Parameter, AstrometryObservation, FisherForecast
from theory.units import *


class MassFunctionKink:
    def __init__(self, A_s=2.105 / 1e9, n_s=0.9665, gen_file='/Users/smsharma/PycharmProjects/Lensing-PowerSpectra/theory/arrays/pk/generate_Pk_kink.py'):
        self.A_s = A_s
        self.n_s = n_s
        self.gen_file = gen_file

    def get_CLASS_kink(self, k_B=0.1, n_B=0.9665, k_max=1000):
        common_settings = {  # Background parameters
            'H0': 67.32117,
            'omega_b': 0.02238280,
            'N_ur': 2.03066666667,
            'omega_cdm': 0.1201075,
            'N_ncdm': 1,
            'omega_ncdm': 0.0006451439,
            'YHe': 0.2454006,
            'tau_reio': 0.05430842,
            'modes': 's',

            # Output settings
            'output': 'mPk',
            'P_k_max_1/Mpc': k_max,
            'k_per_decade_for_pk': 5,
            'k_per_decade_for_bao': 20,
            'P_k_ini type': 'external_Pk',
            'command': "python " + str(self.gen_file),
            'custom1': 0.05,
            'custom2': self.A_s,
            'custom3': self.n_s,
            'custom4': k_B,
            'custom5': n_B,
            'custom6': 1.5 * k_max
        }

        CLASS_inst = Class()
        CLASS_inst.set(common_settings)
        CLASS_inst.compute()

        return CLASS_inst

    def randomword(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))


class Sigma():
    def __init__(self, log10_P_interp):
        self.log10_P_interp = log10_P_interp

    def c200_zcoll(self, M200, f=0.02, C=100., n_iter_max=100):

        c200_ary = np.zeros(n_iter_max)
        z_coll_ary = np.zeros(n_iter_max)

        z_coll_ary[0] = 1.
        c200_ary[0] = 2000.

        for i in range(1, n_iter_max):
            z_coll_ary[i] = fsolve(lambda z_coll: np.abs(
                erfc((self.delta_sc(z_coll) - self.delta_sc(0)) / (np.sqrt(2 * (self.sigma(f * M200) ** 2 - self.sigma(M200) ** 2)))) - (
                    (-1 + np.log(4)) / 2. / (-1 + 1 / (1 + c200_ary[i - 1]) + np.log(1 + c200_ary[i - 1])))),
                z_coll_ary[i-1])[0]
            c200_ary[i] = fsolve(lambda c200: np.abs(
                C * (Planck15.H(z_coll_ary[i]).value / Planck15.H0.value) ** 2 - (
                    200. / 3. / 4. * c200 ** 3 / (np.log(1 + c200) - c200 / (1 + c200)))), c200_ary[i-1])

            z_err = np.abs(
                (z_coll_ary[i] - z_coll_ary[i - 1]) / z_coll_ary[i - 1])
            c200_err = np.abs(
                (c200_ary[i] - c200_ary[i - 1]) / c200_ary[i - 1])

            if z_err < 0.01 and c200_err < 0.01:
                break

        return c200_ary[i], z_coll_ary[i]

    def W(self, k, R):
        """ Top-hat window function for smoothing
        """
        if k * R < 1e-3:
            return 1.
        else:
            return 3 * (k * R) ** -3 * (np.sin(k * R) - (k * R) * np.cos(k * R))

    def log_integrand(self, lnk, R):
        """ Integrand for mass variance sigma
        """
        k = np.exp(lnk)
        return (k ** 3 / (2 * np.pi ** 2)) * 10 ** self.log10_P_interp(np.log10(k * h)) * h ** 3 * self.W(k, R) ** 2

    def sigma_quad(self, R, kmax=15.):
        return np.sqrt(
            quad(lambda lnk: self.log_integrand(lnk, R), np.log(10 ** -6.), np.log(10 ** kmax), epsabs=0.0, epsrel=1e-2,
                 limit=100))[0]

    def sigma(self, M200):
        R = (M200 / (4 / 3. * np.pi * 200 * rho_c)) ** (1 / 3.)
        return self.sigma_quad(R / (Mpc / h))

    def D(self, z):
        ''' Equation 5 of arxiv.org/pdf/1309.5385.pdf
        '''
        gamma = 0.55
        def f(z): return 1 / (1 + z) * (Planck15.Om(z)) ** gamma
        growth_factor = np.exp(-quad(f, 0, z, epsabs=0, epsrel=1e-10)[0])
        return growth_factor

    def delta_sc(self, z):
        return delta_c / self.D(z)
