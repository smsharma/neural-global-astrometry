import sys
sys.path.append("../")
sys.path.append("../../")

from simulation.astrometry_sim import QuasarSim
from theory.units import *
import argparse


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imc", action="store", dest="imc", default=0, type=int)

results = parser.parse_args()

max_sep = 20
nside = 128
lmax = 3*nside - 1
imc = results.imc

sim = QuasarSim(max_sep=max_sep,
                verbose=False,
                sim_uniform=False,
                nside=nside,
                calc_powerspecs=False,
                do_alpha=False,
                save=True,
                save_dir='/mnt/hepheno/smsharma/QuasarSim/',  # '/scratch/sm8383/QuasarSim',
                save_tag='gaussian_quasarsDR2_f0p2_M1e8_R1e2_nside128_sep20_mc' + \
                str(imc),
                sh_profile='Gaussian',
                f_sub=0.2,
                R0=100 * pc)

sim.set_mass_distribution(sim.rho_M_SI, M_min=1e8*M_s, M_max=1e10*M_s,
                          M_min_calib=1e8*M_s, M_max_calib=1e10*M_s, N_calib=150, alpha=-1.9)
sim.set_radial_distribution(sim.r2rho_V_ein_EAQ, R_min=1e-3*kpc, R_max=260*kpc)
sim.set_subhalo_properties(sim.c200_SCP, distdep=False)

sim.analysis_pipeline()
