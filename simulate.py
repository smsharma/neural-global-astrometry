from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
import argparse
import logging
import warnings

sys.path.append("./")
sys.path.append("../")

import numpy as np
import healpy as hp
import torch
from tqdm.auto import tqdm
from astropy.utils.exceptions import AstropyDeprecationWarning

from simulation.astrometry_sim import QuasarSim
from sbi import utils
from theory.units import *

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)


def simulate(n=1000, nside=64, max_sep=20):
    """ High-level simulation script
    """

    logger.info("Generating training data with %s maps", n)

    # Dict to save results
    results = {}

    # Generate points for parameter of interest, here calibrated number of subhalos between 10^8 and 10^10 M_s
    prior = utils.BoxUniform(low=torch.tensor([0.001]), high=torch.tensor([300.0]))
    thetas = prior.sample((n,))

    logger.info("Generating maps...")

    max_sep = 25  # Maximum distance from subhalo center at which to express lensing effects

    # Instantiate simulator
    sim = QuasarSim(max_sep=max_sep, verbose=True, sim_uniform=True, nside=nside, calc_powerspecs=False, do_alpha=False, sh_profile="NFW")

    x = np.zeros((n, 2, hp.nside2npix(nside)))

    # Simulate sample for each parameter point
    for i, theta in tqdm(enumerate(thetas), total=n):

        sim.set_mass_distribution(sim.rho_M_SI, M_min=1e7 * M_s, M_max=1e10 * M_s, M_min_calib=1e8 * M_s, M_max_calib=1.1e10 * M_s, N_calib=theta[0].detach().numpy(), alpha=-1.9)
        sim.set_radial_distribution(sim.r2rho_V_ein_EAQ, R_min=1e-3 * kpc, R_max=260 * kpc)
        sim.set_subhalo_properties(sim.c200_SCP, distdep=False)
        sim.analysis_pipeline(get_sample=True)

        sim.mu_qsrs = hp.reorder(np.transpose(sim.mu_qsrs), r2n=True)  # Convert to NESTED ordering

        x[i, :, :] = 1e6 * sim.mu_qsrs

    results["x"] = x
    results["theta"] = thetas

    return results


def save(data_dir, name, data):
    """ Save simulated data to file
    """

    logger.info("Saving results with name %s", name)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/data".format(data_dir)):
        os.mkdir("{}/data".format(data_dir))
    if not os.path.exists("{}/data/samples".format(data_dir)):
        os.mkdir("{}/data/samples".format(data_dir))

    for key, value in data.items():
        np.save("{}/data/samples/{}_{}.npy".format(data_dir, key, name), value)


def parse_args():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(description="Main high-level script that starts the GCE simulations")

    parser.add_argument(
        "-n", type=int, default=10000, help="Number of samples to generate",
    )
    parser.add_argument("--name", type=str, default=None, help="Sample name")
    parser.add_argument("--dir", type=str, default=".", help="Base directory")
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    name = "train" if args.name is None else args.name
    results = simulate(n=args.n)
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
