import numpy as np
import numpy.typing as npt
import torch
import pyro
import pyro.distributions as dist
from src.models.solid_earth_utils import compute_love_numbers, greens_function


def premis_model(m, x=1, n=128, obs=None) -> np.ndarray:
    """Probabilistic model for uplift rates. This is the main model.
    Args:
        m (np.ndarray): mass time-series data
        s (int, optional): number of stations. Defaults to 2.
        n (int, optional): spherical harmonic degree. Defaults to 128.
        obs (_type_, optional): observed gnss station uplift rates. Defaults to None.

    Returns:
        np.ndarray: predicted uplift
    """

    #  could add dists over mu and sigmas
    #  sigma = pyro.sample("sigma", dist.HalfCauchy(5.))

    # Draw lithospheric elasticity
    E_L = pyro.sample("E_L", dist.Normal(0.0, 1.0))

    # Draw upper mantle elasticity
    E_UM = pyro.sample("E_UM", dist.Normal(0.0, 1.0))

    # Draw lower mantle elasticity
    E_LM = pyro.sample("E_LM", dist.Normal(0.0, 1.0))

    # Draw mass change with time
    # m = pyro.sample("m", dist.Normal(torch.zeros(128), torch.ones(128)))

    sigma_w = pyro.sample("sigma", dist.HalfCauchy(0.1))
    sigma_gf = 1

    station_coordinates = [68.58700000, -33.05270000]  # [lat, lon]
    glacier_coordinates = [68.645056, -33.029416]  # [lat, lon]

    with pyro.plate("data"):
        # Draw Love Numbers
        hlove, nlove = compute_love_numbers()

        # Draw Greens Function
        gf = pyro.sample(
            "gf",
            dist.Normal(
                greens_function(hlove, nlove, glacier_coordinates, station_coordinates),
                sigma_gf,
            ),
        )

        # Draw target
        w = pyro.sample("w", dist.Normal(gf * m, sigma_w), obs=obs)

    return w


def taboo(j: int, e_l: float, e_um: float, e_lm: float) -> np.ndarray:
    """temporary taboo function for running pyro model
    Args:
        j (int): _description_
        e_l (float): _description_
        e_um (float): _description_
        e_lm (float): _description_

    Returns:
        np.ndarray: _description_
    """

    love_numbers = np.random.randint(j)
    return love_numbers


# if __name__ == '__main__':
#     # X = load data
#     # obs = ob
#     _ = premis_train()
