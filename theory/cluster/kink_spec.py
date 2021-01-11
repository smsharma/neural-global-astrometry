import sys

sys.path.append("../")
sys.path.append("../../")
import argparse

import numpy as np
from tqdm import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from theory.spec_calc import PowerSpectra, PowerSpectraPopulations
from theory.kink import MassFunctionKink, Sigma
from theory.units import *
from theory.astrometry_forecast import Parameter, AstrometryObservation, FisherForecast

sys.path.append("/group/hepheno/smsharma/heptools/colossus/")

from colossus.cosmology import cosmology
from colossus.lss import mass_function

# Command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--nB", action="store", dest="nB", default=1.0, type=float)
parser.add_argument("--kB", action="store", dest="kB", default=1, type=float)
parser.add_argument("--Mmin", action="store", dest="Mmin", default=1e4, type=float)
parser.add_argument("--save_tag", action="store", dest="save_tag", default="calib", type=str)
parser.add_argument("--l_cutoff", action="store", dest="l_cutoff", default=-1, type=float)
parser.add_argument("--l_max", action="store", dest="l_max", default=10000, type=int)

results = parser.parse_args()

nB = results.nB
kB = results.kB
Mmin = results.Mmin
save_tag = results.save_tag
l_cutoff = results.l_cutoff
l_max = results.l_max

# pk_dir = '/group/hepheno/smsharma/Lensing-PowerSpectra/theory/arrays/pk/'
# save_dir  = '/group/hepheno/smsharma/Lensing-PowerSpectra/theory/cluster/cluster_out/'

pk_dir = "/home/sm8383/Lensing-PowerSpectra/data/arrays/pk/"
save_dir = "/home/sm8383/Lensing-PowerSpectra/theory/cluster/cluster_out/"

# Get class instance with custom primordial power spectrum

mfk = MassFunctionKink(gen_file=pk_dir + "generate_Pk_kink.py")

CLASS_inst = mfk.get_CLASS_kink(k_B=kB, n_B=nB, k_max=5e2)
CLASS_inst_vanilla = mfk.get_CLASS_kink(k_B=kB, n_B=0.9665, k_max=5e2)

for idnx, inst in enumerate([CLASS_inst_vanilla, CLASS_inst]):
    k_ary = np.logspace(-6, np.log10(5e2), 10000)
    Pk_ary = np.array([inst.pk_lin(k, 0) for k in k_ary])

    log10_k_interp_ary = np.linspace(-6, 20, 10000)
    log10_P_interp = interp1d(np.log10(k_ary / h), np.log10(Pk_ary * h ** 3), bounds_error=False, fill_value="extrapolate")
    log10_P_interp_ary = (log10_P_interp)(log10_k_interp_ary)

    if idnx == 1:
        file_kinked = pk_dir + "pk" + str(kB) + "_" + str(nB) + "_" + str(Mmin) + ".dat"
        np.savetxt(file_kinked, np.transpose([log10_k_interp_ary, log10_P_interp_ary]), delimiter="\t")
    else:
        file_base = pk_dir + "pk" + str(kB) + "_" + str(nB) + "_" + str(Mmin) + "_base.dat"
        np.savetxt(file_base, np.transpose([log10_k_interp_ary, log10_P_interp_ary]), delimiter="\t")

# Get mass function from colossus

cosmo = cosmology.setCosmology("planck18")

M_ary = np.logspace(4, 12, 50)

dndlnM_vanilla_ary = mass_function.massFunction(M_ary, 0.0, mdef="200m", model="tinker08", q_in="M", q_out="dndlnM", ps_args={"model": mfk.randomword(10), "path": file_base})
dndlnM_ary = mass_function.massFunction(M_ary, 0.0, mdef="200m", model="tinker08", q_in="M", q_out="dndlnM", ps_args={"model": mfk.randomword(10), "path": file_kinked})

dndlnM_vanilla_interp = interp1d(np.log10(M_ary * M_s), np.log10(dndlnM_vanilla_ary / M_ary))
dndlnM_interp = interp1d(np.log10(M_ary * M_s), np.log10(dndlnM_ary / M_ary), fill_value="extrapolate")

# Calibrate to number density at high masses
N_calib = 150.0

pref = N_calib / quad(lambda M: 10 ** dndlnM_vanilla_interp(np.log10(M)), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-4)[0]
N_calib_new = pref * quad(lambda M: 10 ** dndlnM_interp(np.log10(M)), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-4)[0]

# Get c200

sig = Sigma(log10_P_interp)

M_ary_conc = np.logspace(8, 13, 10) * M_s
c200_ary = [sig.c200_zcoll(M, C=100.0, f=0.02)[0] for M in tqdm(M_ary_conc)]

c200_interp = interp1d(np.log10(M_ary_conc), np.log10(c200_ary), bounds_error=False, fill_value="extrapolate")


def dndM(M):
    return 10 ** dndlnM_interp(np.log10(M))


def c200_custom(M):
    return 10 ** c200_interp(np.log10(M))


pspecpop = PowerSpectraPopulations(l_max=l_max)

pspecpop.set_radial_distribution(pspecpop.r2rho_V_NFW, R_min=1e-2 * kpc, R_max=260 * kpc)
pspecpop.set_mass_distribution(dndM, M_min=Mmin * M_s, M_max=0.01 * 1.1e12 * M_s, M_min_calib=1e8 * M_s, M_max_calib=1e10 * M_s, N_calib=N_calib_new)
pspecpop.set_subhalo_properties(c200_custom)

if l_cutoff == -1.0:
    l_cutoff = pspecpop.l_cutoff
else:
    l_cutoff = l_cutoff * pc

pspecpop.get_C_l_total_ary(l_los_min=l_cutoff)
C_l_mu = pspecpop.C_l_calc_ary

pspecpop.get_C_l_total_ary(l_los_min=l_cutoff, accel=True)
C_l_alpha = pspecpop.C_l_calc_ary

l_ary = pspecpop.l_ary_calc


def get_sens(Cl_ary, Cl_ary_accel, f_sky=0.05, sigma_mu=10000.0, N_q_mu=1e8, sigma_alpha=1, N_q_alpha=1e12, l_min_mu=10, l_max_mu=2000, l_min_alpha=50, l_max_alpha=10000, l_max=10000):
    fDM_base = 1
    dfDM_base = 0.1

    Cl_ary_fid = np.array(Cl_ary) * fDM_base
    Cl_ary_accel_fid = np.array(Cl_ary_accel) * fDM_base

    p = np.array(Cl_ary) * (fDM_base + dfDM_base)
    m = np.array(Cl_ary) * (fDM_base - dfDM_base)

    p_a = np.array(Cl_ary_accel) * (fDM_base + dfDM_base)
    m_a = np.array(Cl_ary_accel) * (fDM_base - dfDM_base)

    fDM = Parameter("fDM", fDM_base, dfDM_base, None, True, p, m, p_a, m_a, l_min=1, l_max=l_max)

    if sigma_mu == -1:
        Cl_ary_fid = None
    if sigma_alpha == -1:
        Cl_ary_accel_fid = None

    parameters = [Cl_ary_fid, Cl_ary_accel_fid, 1, l_max, fDM]

    observation = AstrometryObservation(fsky=f_sky, sigma_mu=sigma_mu, sigma_alpha=sigma_alpha, N_q_mu=N_q_mu, N_q_alpha=N_q_alpha, l_min_mu=l_min_mu, l_max_mu=l_max_mu, l_min_alpha=l_min_alpha, l_max_alpha=l_max_alpha)

    fshr = FisherForecast(parameters, observation)
    lim = 1.64 * np.sqrt(np.linalg.inv(fshr.fshr_cls + fshr.fshr_prior)[0, 0])
    return lim, fDM_base / fshr.pars_vary[0].sigma


l_ary_arange = np.arange(np.min(l_ary), np.max(l_ary))
C_l_mu = 10 ** np.interp(np.log10(l_ary_arange), np.log10(l_ary), np.log10(C_l_mu))
C_l_alpha = 10 ** np.interp(np.log10(l_ary_arange), np.log10(l_ary), np.log10(C_l_alpha))

lim_ska, sig_ska = get_sens(C_l_mu, C_l_alpha, f_sky=1, sigma_mu=1, N_q_mu=1e8, sigma_alpha=-1, l_max=l_max, l_max_alpha=3, l_max_mu=5000)
lim_wfirst, sig_wfirst = get_sens(C_l_mu, C_l_alpha, f_sky=0.05, sigma_mu=-1, N_q_mu=1e8, sigma_alpha=0.1, l_max=l_max, l_max_alpha=500000, l_max_mu=3)
# lim_gaia, sig_gaia = get_sens(C_l_mu, C_l_alpha, f_sky=0.05, sigma_mu=-1, N_q_mu=1e8, sigma_alpha=10, l_max=l_max, l_max_alpha=500000, l_max_mu=3)
lim_gaia, sig_gaia = get_sens(C_l_mu, C_l_alpha, f_sky=0.05, sigma_mu=-1, N_q_mu=1e8, N_q_alpha=2e9 * 20, sigma_alpha=2, l_max=l_max, l_max_alpha=50000, l_max_mu=3)

np.savez(save_dir + "/" + save_tag + "_" + str(kB) + "_" + str(nB) + "_" + str(Mmin) + ".npz", C_l_mu=C_l_mu, C_l_alpha=C_l_alpha, l_ary=l_ary, dndlnM_ary=dndlnM_ary, M_ary=M_ary, c200_ary=c200_ary, M_ary_conc=M_ary_conc / M_s, sig_ska=np.array([lim_ska, sig_ska]), sig_wfirst=np.array([lim_wfirst, sig_wfirst]), sig_gaia=np.array([lim_gaia, sig_gaia]))
