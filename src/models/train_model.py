import numpy as np
import numpy.typing as npt
from pyparsing import javaStyleComment
import torch
import pyro
import pyro.distributions as dist
from TABOO import taboo
import greens_function


def premis_train(X, x=2, n=128, obs=None) -> np.ndarray:
    """
    Args:
        X (_type_): _description_
        x (int, optional): _description_. Defaults to 2.
        n (int, optional): _description_. Defaults to 128.
        obs (_type_, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """

    #  could add dists over mu and sigmas
    #  sigma = pyro.sample("sigma", dist.HalfCauchy(5.))
    #
    # Draw lithospheric elasticity
    E_L = pyro.sample("E_L", dist.Normal(0.0, 1.0))

    # Draw upper mantle elasticity
    E_UM = pyro.sample("E_UM", dist.Normal(0.0, 1.0))

    # Draw lower mantle elasticity
    E_LM = pyro.sample("E_LM", dist.Normal(0.0, 1.0))

    # Draw mass change with time
    m = pyro.sample("m", dist.Normal(torch.zeros(128), torch.ones(128)))

    sigma_w = pyro.sample("sigma", dist.HalfCauchy(0.1))
    sigma_GF = 1

    A = 1
    c = 2 * range(n) + 1
    D = np.random.randn(n)

    with pyro.plate("data"):
        # Draw Love Numbers
        LN = torch.tensor(
            taboo(j=n, e_l=E_L.item(), e_um=E_UM.item(), e_lm=E_LM.item())
        )

        # Draw Greens Funciton
        GF = pyro.sample("GF", dist.Normal(greens_function(A, D, LN), sigma_GF))

        # Draw target
        w = pyro.sample("w", dist.Normal(GF @ m, sigma_w), obs=obs)

    return w


def taboo(j: int, e_l: float, e_um: float, e_lm: float) -> npt.NDArray[np.int_]:
    return np.random.randint(j)
