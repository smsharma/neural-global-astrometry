import numpy as np
from theory.units import *


class FisherForecast:
    def __init__(self, parameters, observation):

        self.c_l_mu_fid, self.c_l_alpha_fid, self.l_min, self.l_max, * \
            self.parameters = parameters
        self.observation = observation
        self.n_pars_tot = 0
        self.n_pars_vary = 0

        self.setup_pars()
        self.setup_tracers()
        self.get_fisher()
        self.get_sigmas()

    def setup_pars(self):

        self.l_ary = np.arange(self.l_min, self.l_max)

        self.n_pars_tot = len(self.parameters)

        self.pars_vary = []
        for par in self.parameters:
            if par.vary:
                self.n_pars_vary += 1
                self.pars_vary.append(par)

    def setup_tracers(self):

        self.n_tracers = 0
        self.l_min_arr = []
        self.l_max_arr = []
        self.c_l_fid_ary = []
        self.c_l_p_ary = []
        self.c_l_m_ary = []
        self.N_l_ary = []

        if self.observation.has_mu:
            self.n_tracers += 1
            self.l_min_arr.append(self.observation.l_min_mu)
            self.l_max_arr.append(self.observation.l_max_mu)
            self.c_l_fid_ary.append(self.c_l_mu_fid)
            self.N_l_ary.append(np.ones_like(self.l_ary) *
                                self.observation.N_l_mu_val)
            self.c_l_p_ary.append([par.C_l_mu_p for par in self.pars_vary])
            self.c_l_m_ary.append([par.C_l_mu_m for par in self.pars_vary])

        if self.observation.has_alpha:
            self.n_tracers += 1
            self.l_min_arr.append(self.observation.l_min_alpha)
            self.l_max_arr.append(self.observation.l_max_alpha)
            self.c_l_fid_ary.append(self.c_l_alpha_fid)
            self.N_l_ary.append(np.ones_like(self.l_ary) *
                                self.observation.N_l_alpha_val)
            self.c_l_p_ary.append([par.C_l_alpha_p for par in self.pars_vary])
            self.c_l_m_ary.append([par.C_l_alpha_m for par in self.pars_vary])

        self.cl_fid_ary = np.zeros(
            [len(self.l_ary), self.n_tracers, self.n_tracers])
        self.cl_noise_ary = np.zeros(
            [len(self.l_ary), self.n_tracers, self.n_tracers])
        self.dcldpar_ary = np.zeros([self.n_pars_vary, len(
            self.l_ary), self.n_tracers, self.n_tracers])

        for itr in range(self.n_tracers):
            self.cl_fid_ary[:, itr, itr] = self.c_l_fid_ary[itr]
            self.cl_noise_ary[:, itr, itr] = self.N_l_ary[itr]

            for ipar, par in enumerate(self.pars_vary):
                self.dcldpar_ary[ipar, :, itr, itr] = (self.c_l_p_ary[itr][ipar] - self.c_l_m_ary[itr][ipar]) / (
                    2 * par.dpar)

    def get_fisher(self):

        self.fshr_l = np.zeros(
            [self.n_pars_vary, self.n_pars_vary, len(self.l_ary)])
        self.fshr_prior = np.zeros([self.n_pars_vary, self.n_pars_vary])
        self.fshr_cls = np.zeros([self.n_pars_vary, self.n_pars_vary])

        for ipar, par in enumerate(self.pars_vary):
            if par.prior is not None:
                self.fshr_prior[ipar, ipar] = 1 / par.prior ** 2

        for il, ell in enumerate(self.l_ary):
            indices = np.where((self.l_min_arr <= ell) &
                               (self.l_max_arr >= ell))[0]
            cl_fid = self.cl_fid_ary[il, indices, :][:, indices]
            cl_noise = self.cl_noise_ary[il, indices, :][:, indices]
            icl = np.linalg.inv(cl_fid + cl_noise)
            for i in np.arange(self.n_pars_vary):
                dcl1 = self.dcldpar_ary[i, il, indices, :][:, indices]
                for j in np.arange(self.n_pars_vary - i) + i:
                    dcl2 = self.dcldpar_ary[j, il, indices, :][:, indices]
                    self.fshr_l[i, j, il] = self.observation.fsky * (ell + 0.5) * np.trace(
                        np.dot(dcl1, np.dot(icl, np.dot(dcl2, icl))))
                    if i != j:
                        self.fshr_l[j, i, il] = self.fshr_l[i, j, il]

        self.fshr_cls[:, :] = np.sum(np.nan_to_num(self.fshr_l), axis=2)

    def get_sigmas(self):
        fshr = self.fshr_cls + self.fshr_prior
        covar = np.linalg.inv(fshr)
        for ipar, par in enumerate(self.pars_vary):
            par.sigma = np.sqrt(covar[ipar, ipar])


class Parameter:
    def __init__(self, name='fDM',
                 fid_val=0.2, dpar=0.02, prior=None, vary=True,
                 C_l_mu_p=None, C_l_mu_m=None,
                 C_l_alpha_p=None, C_l_alpha_m=None,
                 l_min=1, l_max=1000):
        self.name = name
        self.fid_val = fid_val
        self.dpar = dpar
        self.prior = prior
        self.vary = vary

        self.C_l_mu_p = C_l_mu_p
        self.C_l_mu_m = C_l_mu_m

        self.C_l_alpha_p = C_l_alpha_p
        self.C_l_alpha_m = C_l_alpha_m

        self.l_min = l_min
        self.l_max = l_max


class AstrometryObservation:
    def __init__(self, fsky=1., sigma_mu=1, sigma_alpha=None,
                 N_q_mu=1e9, N_q_alpha=1e10,
                 l_min_mu=1, l_max_mu=1000,
                 l_min_alpha=1, l_max_alpha=1000):
        self.fsky = fsky

        self.l_min_mu = l_min_mu
        self.l_max_mu = l_max_mu

        self.l_min_alpha = l_min_alpha
        self.l_max_alpha = l_max_alpha

        self.l_ary_mu = np.arange(l_min_mu, l_max_mu)
        self.l_ary_alpha = np.arange(l_min_alpha, l_max_alpha)

        self.sigma_mu = sigma_mu
        self.sigma_alpha = sigma_alpha

        self.has_mu = False
        self.has_alpha = False

        if self.sigma_mu is not None:
            self.has_mu = True
            self.N_l_mu_val = 4 * np.pi * sigma_mu ** 2 / N_q_mu
        if self.sigma_alpha is not None:
            self.has_alpha = True
            self.N_l_alpha_val = 4 * np.pi * sigma_alpha ** 2 / N_q_alpha
