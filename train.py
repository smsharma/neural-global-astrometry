#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import json
import logging
import argparse

sys.path.append("./")

import numpy as np
import healpy as hp
import torch
from pytorch_lightning.loggers import MLFlowLogger
import mlflow

from models.embedding import SphericalGraphCNN
from simulation import create_mask as cm
from sbi import utils
from sbi.inference import RatioEstimator


def train(data_dir, experiment_name, sample_name, nside_max=64, kernel_size=4, laplacian_type="combinatorial", n_neighbours=8, batch_size=256, max_num_epochs=50, stop_after_epochs=10, clip_max_norm=1.0, validation_fraction=0.15, initial_lr=1e-3, device=None, optimizer=torch.optim.AdamW, optimizer_kwargs={"weight_decay": 1e-5}, activation="relu", conv_source="deepsphere", conv_type="chebconv", num_samples=None, sigma_noise=0.0022, fc_dims=[[-1, 128], [128, 128], [128, 64]], truncate_conv=None, numpy_noise=False, pooling_end="average", b_mask=None, sigma_noise_model_file=None):

    # Cache hyperparameters to log
    params_to_log = locals()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("")
    logging.info("Creating estimator")
    logging.info("")

    # Get mask of central pixel for nside=1
    hp_mask_nside1 = cm.make_mask_total(nside=1, mask_ring=False)

    indexes_list = []
    masks_list = []

    assert (nside_max & (nside_max - 1) == 0) and nside_max != 0, "Invalid nside"

    nside_list = [int(nside_max / (2 ** i)) for i in np.arange(hp.nside2order(nside_max))]

    # Build indexes corresponding to subsequent nsides
    for nside in nside_list:
        hp_mask = hp.ud_grade(hp_mask_nside1, nside)
        hp_mask = hp.reorder(hp_mask, r2n=True)  # Switch to NESTED pixel order as that's required for DeepSphere batchnorm
        masks_list.append(hp_mask)
        indexes_list.append(np.arange(hp.nside2npix(nside))[~hp_mask])

    hp_mask_nside1 = hp.reorder(hp_mask_nside1, r2n=True)  # Switch to NESTED pixel order as that's required for DeepSphere batchnorm

    # # ROI mask mimicking partial-sky coverage
    # if b_mask is not None:
    #     logging.info("Applying a latitude mask at {} deg".format(b_mask))
    #     mask_roi = ~cm.make_mask_total(band_mask=True, band_mask_range=b_mask, nside=nside_max)
    #     mask_roi = hp.reorder(mask_roi, r2n=True)
    #     mask_roi = torch.Tensor(mask_roi).bool()
    # else:
    #     mask_roi = None

    # Construct anisotropic noise model from saved map
    if sigma_noise_model_file is not None:
        logging.info("Applying external noise model {}".format(sigma_noise_model_file))
        sigma_noise_model = np.load("{}/{}".format(data_dir, sigma_noise_model_file))
        mask_roi = sigma_noise_model[0] == 0
        sigma_noise_model[:, ~mask_roi] /= np.sqrt(np.expand_dims(np.mean(sigma_noise_model[:, ~mask_roi] ** 2, axis=1), axis=1) / sigma_noise ** 2)
        sigma_noise_model = hp.ud_grade(sigma_noise_model, nside_out=nside_max, power=0)
        sigma_noise_model = hp.reorder(sigma_noise_model, r2n=True)

        mask_roi = sigma_noise_model[0] == 0

        mask_roi = torch.Tensor(mask_roi).bool()
        sigma_noise = torch.Tensor(sigma_noise_model).unsqueeze((0)).to(device)
    else:
        mask_roi = None

    # Whether to truncate convolutional layers
    if truncate_conv is not None:
        indexes_list = indexes_list[:truncate_conv]
        nside_list = nside_list[:truncate_conv]

    # Parameter of interest, here calibrated number of subhalos between 10^8 and 10^10 M_s
    prior = utils.BoxUniform(low=torch.tensor([0.001]), high=torch.tensor([300.0]))

    # MLFlow logger
    tracking_uri = "file:{}/logs/mlruns".format(data_dir)
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    mlf_logger.log_hyperparams(params_to_log)

    # Specify datasets
    x_filename = "{}/samples/x_{}.npy".format(data_dir, sample_name)
    theta_filename = "{}/samples/theta_{}.npy".format(data_dir, sample_name)

    # Embedding net (feature extractor)
    sg_embed = SphericalGraphCNN(nside_list, indexes_list, kernel_size=kernel_size, laplacian_type=laplacian_type, n_params=1, activation=activation, conv_source=conv_source, conv_type=conv_type, in_ch=2, n_neighbours=n_neighbours, fc_dims=fc_dims, pooling_end=pooling_end, mask=mask_roi)

    # Instantiate the parameterized classifier
    neural_classifier = utils.classifier_nn(model="mlp_mixed", embedding_net_x=sg_embed, sigma_noise=sigma_noise)

    # Setup the inference procedure with
    posterior_estimator = RatioEstimator(prior=prior, classifier=neural_classifier, show_progress_bars=True, logging_level="INFO", device=device.type, summary_writer=mlf_logger)

    # Model training
    density_estimator = posterior_estimator.train(x=x_filename, theta=theta_filename, num_samples=num_samples, proposal=prior, training_batch_size=batch_size, max_num_epochs=max_num_epochs, stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm, validation_fraction=validation_fraction, initial_lr=initial_lr, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, numpy_noise=numpy_noise)

    # Save density estimator
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.pytorch.log_model(density_estimator, "density_estimator")

    # Check to make sure model can be succesfully loaded
    model_uri = "runs:/{}/density_estimator".format(mlf_logger.run_id)
    density_estimator = mlflow.pytorch.load_model(model_uri)
    posterior = posterior_estimator.build_posterior(density_estimator)


def parse_args():
    parser = argparse.ArgumentParser(description="High-level script for the training of the neural likelihood ratio estimators")

    # Main options
    parser.add_argument("--sample", type=str, help="Sample name")
    parser.add_argument("--name", type=str, default="test", help="Experiment name to save with MLflow")
    parser.add_argument("--laplacian_type", type=str, default="combinatorial", help='"normalized" or "combinatorial" Laplacian')
    parser.add_argument("--conv_source", type=str, default="deepsphere", help='Use "deepsphere" or "geometric" implementation of ChebConv layer')
    parser.add_argument("--conv_type", type=str, default="chebconv", help='Use "chebconv" or "gcn" graph convolution layers')
    parser.add_argument("--n_neighbours", type=int, default=8, help="Number of neighbours in graph")
    parser.add_argument("--activation", type=str, default="relu", help='Nonlinearity, "relu" or "selu"')
    parser.add_argument("--max_num_epochs", type=int, default=50, help="Max number of training epochs")
    parser.add_argument("--kernel_size", type=int, default=5, help="GNN  kernel size")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to load")
    parser.add_argument("--numpy_noise", type=int, default=0, help="Whether to add noise using numpy (while loading) or torch (in model)")
    parser.add_argument("--sigma_noise", type=float, default=0.0023, help="Noise added to dataset")
    parser.add_argument("--dir", type=str, default=".", help="Head directory containing 'data/samples' and 'data/models' sub-directories")
    parser.add_argument("--fc_dims", type=str, default="[[-1, 1024],[1024, 256]]", help="Specification of fully-connected classifier layers")
    parser.add_argument("--pooling_end", type=str, default="average", help="'average' or 'flatten' final spherical feature map")
    # parser.add_argument("--b_mask", type=float, default=0.0, help="Partial-sky band latitude")
    parser.add_argument("--initial_lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--sigma_noise_model_file", type=str, default="None", help="File name for anisotropic noise model")

    # Training option
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.INFO,
    )
    logging.info("Hi!")

    args = parse_args()

    if args.num_samples == -1:
        args.num_samples = None

    if args.sigma_noise_model_file == "None":
        args.sigma_noise_model_file = None

    if args.fc_dims == "None":
        args.fc_dims = None
    else:
        args.fc_dims = list(json.loads(args.fc_dims))

    train(data_dir="{}/data/".format(args.dir), sample_name=args.sample, experiment_name=args.name, batch_size=args.batch_size, activation=args.activation, kernel_size=args.kernel_size, max_num_epochs=args.max_num_epochs, laplacian_type=args.laplacian_type, conv_source=args.conv_source, conv_type=args.conv_type, n_neighbours=args.n_neighbours, num_samples=args.num_samples, sigma_noise=args.sigma_noise, fc_dims=args.fc_dims, numpy_noise=args.numpy_noise, pooling_end=args.pooling_end, sigma_noise_model_file=args.sigma_noise_model_file, initial_lr=args.initial_lr)  # , b_mask=args.b_mask

    logging.info("All done! Have a nice day!")
