# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import (
    Any,
    Dict,
    Optional,
)
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils import del_entries


class RatioBasedPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNRE.<br/><br/>
    SNRE trains a neural network to approximate likelihood ratios, which in turn can be
    used obtain an unnormalized posterior $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$. The `SNRE_Posterior` class wraps the trained network such that one can
    directly evaluate the unnormalized posterior log-probability $p(\theta|x) \propto
    p(x|\theta) \cdot p(\theta)$ and draw samples from the posterior with
    MCMC. Note that, in the case of single-round SNRE_A / AALR, it is possible to
    evaluate the log-probability of the **normalized** posterior, but sampling still
    requires MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self, method_family: str, neural_net: nn.Module, prior, x_shape: torch.Size, device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            device: Training device, e.g., cpu or cuda:0.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def log_prob(self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,) -> Tensor:
        r"""
        Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This corresponds to an **unnormalized** posterior log-probability. Only for
        single-round SNRE_A / AALR, the returned log-probability will correspond to the
        **normalized** log-probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """

        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        with torch.set_grad_enabled(track_gradients):
            # Send to device for evaluation, send to CPU for comparison with prior.
            log_ratio = self.net(torch.cat((theta.to(self.device), x.to(self.device)), dim=1)).reshape(-1).to("cpu")
            return log_ratio + self.prior.log_prob(theta)
