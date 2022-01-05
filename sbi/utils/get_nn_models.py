# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Callable

from torch import nn

from sbi.neural_nets.classifier import build_mlp_mixed_classifier

# from sbi.neural_nets.flow import build_made, build_maf, build_nsf
# from sbi.neural_nets.mdn import build_mdn


def classifier_nn(
    model: str,
    z_score_theta: bool = True,
    z_score_x: bool = True,
    # hidden_features: int = 50,
    embedding_net_theta: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    sigma_noise: float = 0.0,
) -> Callable:
    r"""
    Returns a function that builds a classifier for learning density ratios.

    This function will usually be used for SNRE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of classifier that will be created. One of [`linear`, `mlp`,
            `resnet`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network.
        hidden_features: Number of hidden features.
        embedding_net_theta:  Optional embedding network for parameters $\theta$.
        embedding_net_x:  Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                # "hidden_features",
                "embedding_net_y",
                "embedding_net_x",
                "sigma_noise",
            ),
            (
                z_score_x,
                z_score_theta,
                # hidden_features,
                embedding_net_x,
                embedding_net_theta,
                sigma_noise,
            ),
        )
    )

    def build_fn(batch_theta, batch_x):
        if model == "mlp_mixed":
            return build_mlp_mixed_classifier(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    return build_fn
