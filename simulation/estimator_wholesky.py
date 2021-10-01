import healpy as hp
from healpy.sphtfunc import Alm
import numpy as np
from tqdm import *


def fB(l, m, gtheta_ary, gphi_ary, lmax):
    """
    From Kostelec, eq. 15, modified for our spherical harmonic convention
    """
    return 1 / np.sqrt(l * (l + 1)) * (l * np.sqrt(((l - m + 1) * (l + m + 1)) / ((2 * l + 3) * (2 * l + 1))) * gtheta_ary[Alm.getidx(lmax, l + 1, m)] - (l + 1) * np.sqrt((l - m) * (l + m) / ((2 * l - 1) * (2 * l + 1))) * gtheta_ary[Alm.getidx(lmax, l - 1, m)] + 1.0j * m * gphi_ary[Alm.getidx(lmax, l, m)])


def fC(l, m, gtheta_ary, gphi_ary, lmax):
    """
    From Kostelec, eq. 15, modified for our spherical harmonic convention
    """
    return 1 / np.sqrt(l * (l + 1)) * (l * np.sqrt(((l - m + 1) * (l + m + 1)) / ((2 * l + 3) * (2 * l + 1))) * gphi_ary[Alm.getidx(lmax, l + 1, m)] - (l + 1) * np.sqrt((l - m) * (l + m) / ((2 * l - 1) * (2 * l + 1))) * gphi_ary[Alm.getidx(lmax, l - 1, m)] - 1.0j * m * gtheta_ary[Alm.getidx(lmax, l, m)])


def get_alm_theta_phi(map_theta, map_phi, map_theta_2=None, map_phi_2=None):
    nside = hp.npix2nside(len(map_theta))
    lmax = 3 * nside - 1
    alm_theta = hp.map2alm(map_theta, lmax=lmax)
    alm_phi = hp.map2alm(map_phi, lmax=lmax)
    return alm_theta, alm_phi, lmax


def get_Cells(alm_theta, alm_phi, lmax):
    fB_ary = np.zeros((lmax, lmax), dtype=np.complex)
    fC_ary = np.zeros((lmax, lmax), dtype=np.complex)
    for l in range(lmax - 1):
        for m in range(l):
            fB_ary[l][m] = fB(l, m, alm_theta, alm_phi, lmax)
            fC_ary[l][m] = fC(l, m, alm_theta, alm_phi, lmax)

    Cl_B = 2 * np.sum(np.abs(fB_ary) ** 2, axis=1)
    Cl_C = 2 * np.sum(np.abs(fC_ary) ** 2, axis=1)

    return Cl_B, Cl_C, fB_ary, fC_ary


def get_vector_alm(map_theta, map_phi):
    # Construct auxiliary maps
    nside = hp.npix2nside(len(map_theta))
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    map_theta_aux = map_theta / np.sin(theta)
    map_phi_aux = map_phi / np.sin(theta)
    alm_theta, alm_phi, lmax = get_alm_theta_phi(map_theta_aux, map_phi_aux)
    Cl_B, Cl_C, fB_ary, fC_ary = get_Cells(alm_theta, alm_phi, lmax)
    return Cl_B, Cl_C, fB_ary, fC_ary


def get_cross_correlation_Cells(flm1, flm2):
    """ Get cross-correlation C_ells from to mu_lm matrices
    """
    return np.sum(np.conjugate(flm1) * flm2 + np.conjugate(flm2) * flm1, axis=1)

