# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from sbi import utils as utils
from sbi.user_input.user_input_checks import process_x
from sbi.utils.torchutils import (
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)


class NeuralPosterior(ABC):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.<br/><br/>
    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior. The neural network itself can be accessed via the `.net`
    attribute.
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
            device: Training device, e.g., cpu or cuda.
        """
        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError("Method family unsupported.")

        self.net = neural_net
        self.device = device
        self.prior = prior

    @abstractmethod
    def log_prob(self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,) -> Tensor:
        """See child classes for docstring."""
        pass

    def _prepare_theta_and_x_for_log_prob_(self, theta: Tensor, x: Optional[Tensor] = None,) -> Tuple[Tensor, Tensor]:
        r"""Returns $\theta$ and $x$ in shape that can be used by posterior.log_prob().

        Checks shapes of $\theta$ and $x$ and then repeats $x$ as often as there were
        batch elements in $\theta$.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.

        Returns:
            ($\theta$, $x$) with the same batch dimension, where $x$ is repeated as
            often as there were batch elements in $\theta$ originally.
        """

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on.
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
        # when calling nflows in order to have matching shapes of theta and context x
        # at neural network evaluation time.
        x = self._match_x_with_theta_batch_shape(x, theta)

        return theta, x
