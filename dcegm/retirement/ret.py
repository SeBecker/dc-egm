"""This module contains all functions that are related to the retirement model."""
import numpy as np
import scipy.interpolate as scin


def util(consumption, working, theta, duw):
    """CRRA utility"""

    u = (consumption ** (1 - theta) - 1) / (1 - theta)
    u = u - duw * (working)

    return u


def mutil(consumption, theta):
    """Marginal utility CRRA"""

    mu = consumption ** (-theta)

    return mu


def imutil(mutil, theta):
    """Inverse marginal utility CRRA
    Consumption as a function of marginal utility"""

    cons = mutil ** (-1 / theta)

    return cons


def income(it, shock, coeffs_age_poly):
    """Income in period t given normal shock"""

    ages = (it + 20) ** np.arange(len(coeffs_age_poly))
    w = np.exp(coeffs_age_poly @ ages + shock)

    return w


def budget(it, savings, shocks, working, num_grid, n_quad_points, r, coeffs_age_poly):
    """Wealth, M_{t+1} in period t+1, where it == t

    Arguments
    ---------
        savings: np.array of savings with length ngridm
        shocks: np.array of shocks with length expn

    Returns
    -------
        w1: matrix with dimension (expn, ngridm) of all possible
    next period wealths
    """

    w1 = np.full(
        (num_grid, n_quad_points), income(it + 1, shocks, coeffs_age_poly) * working
    ).T + np.full((n_quad_points, num_grid), savings * (1 + r))

    return w1


def mbudget(ngridm, n_quad_points, r):
    """Marginal budget:
    Derivative of budget with respect to savings"""

    mw1 = np.full((n_quad_points, ngridm), (1 + r))

    return mw1


def value_function(working, it, x, value, beta, theta, duw):
    x = x.flatten("F")

    res = np.full(x.shape, np.nan)
    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    mask = x < value[it][working][1, 0]

    # Calculate t+1 value function in the constrained region
    res[mask] = util(x[mask], working, theta, duw) + beta * value[it][working][0, 1]

    # Calculate t+1 value function in non-constrained region
    # inter- and extrapolate
    interpolation = scin.interp1d(
        value[it][working][:, 0],
        value[it][working][:, 1],
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
    )
    res[~mask] = interpolation(x[~mask])

    return res


# Calculation of probability to choose work, if a worker today
def choice_probs_worker(x, lambda_):
    """Calculate the probability of choosing work in t+1
    for state worker given t+1 value functions"""

    mx = np.amax(x, axis=0)
    mxx = x - mx
    res = np.exp(mxx[1, :] / lambda_) / np.sum(np.exp(mxx / lambda_), axis=0)

    return res


# Expected value function calculation in state worker
def logsum(x, lambda_):
    """Calculate expected value function"""

    mx = np.amax(x, axis=0)
    mxx = x - mx
    res = mx + lambda_ * np.log(np.sum(np.exp(mxx / lambda_), axis=0))

    return res
