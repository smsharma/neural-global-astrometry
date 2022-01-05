# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import torch
from torch import Tensor, nn

from sbi.utils.sbiutils import standardizing_net


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
    """

    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma
        self.register_buffer("noise", torch.tensor(0))

    def forward(self, x):
        if self.sigma is not None:
            if torch.is_tensor(self.sigma):
                x = x.to(self.sigma.device)
                self.noise = self.noise.to(self.sigma.device)
            sampled_noise = self.noise.expand(*x.size()).detach().float().normal_() * self.sigma
            x = x + sampled_noise
        return x


class StandardizeInputs(nn.Module):
    def __init__(self, embedding_net_x, embedding_net_y, batch_x, batch_y, z_score_x, z_score_y, sigma_noise=None):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y

        self.batch_x = batch_x
        self.batch_y = batch_y

        self.add_noise = GaussianNoise(sigma_noise)

        if z_score_x:
            self.standardizing_net_x = standardizing_net(batch_x)
        else:
            self.standardizing_net_x = nn.Identity()

        if z_score_y:
            self.standardizing_net_y = standardizing_net(self.add_noise(batch_y))
        else:
            self.standardizing_net_y = nn.Identity()

    def forward(self, x, theta, add_noise=True):

        # print(x[x != 0].std())

        if add_noise:
            x = self.add_noise(x)

        # print(x[x != 0].std())

        x = self.standardizing_net_y(x)

        theta = self.standardizing_net_x(theta)
        theta = self.embedding_net_x(theta)

        x = self.embedding_net_y(x, theta)

        return x


class SequentialMulti(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def build_mlp_mixed_classifier(batch_x: Tensor = None, batch_y: Tensor = None, z_score_x: bool = True, z_score_y: bool = True, embedding_net_x: nn.Module = nn.Identity(), embedding_net_y: nn.Module = nn.Identity(), sigma_noise: float = 0.0) -> nn.Module:
    """Builds MLP classifier.

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Neural network.
    """

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_y(batch_y[:1], batch_x[:1]).numel()

    neural_net = nn.Sequential(
        nn.ReLU(),
        nn.Linear(x_numel, 1),
    )

    input_layer = StandardizeInputs(embedding_net_x, embedding_net_y, batch_x, batch_y, z_score_x=z_score_x, z_score_y=z_score_y, sigma_noise=sigma_noise)

    neural_net = SequentialMulti(input_layer, neural_net)

    return neural_net
