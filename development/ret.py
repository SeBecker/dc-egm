import numpy as np

# from numpy.matlib import *
# from scipy.optimize import *


# Functions: utility and budget constraint


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


def budget(it, savings, shocks, working, ngridm, n_quad_points, r):
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

    w1 = np.full((ngridm, n_quad_points), income(it + 1, shocks) * working).T + np.full(
        (n_quad_points, ngridm), savings * (1 + r)
    )

    return w1


def mbudget(ngridm, n_quad_points, r):
    """Marginal budget:
    Derivative of budget with respect to savings"""

    mw1 = np.full((n_quad_points, ngridm), (1 + r))

    return mw1


# Value function for worker
# interpolate and extrapolate are potentially substitutable by the interpolate function below


def value_function(working, it, x, value, beta):
    """Value function calculation for the """

    x = x.flatten("F")

    res = np.full(x.shape, np.nan)

    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    mask = x < value[1, 0, working, it]

    # Calculate t+1 value function in the constrained region
    res[mask] = util(x[mask], working) + beta * value[0, 1, working, it]

    # Calculate t+1 value function in non-constrained region
    # interpolate
    res[~mask] = np.interp(x[~mask], value[:, 0, working, it], value[:, 1, working, it])
    # extrapolate
    slope = (value[-2, 1, working, it] - value[-1, 1, working, it]) / (
        value[-2, 0, working, it] - value[-1, 0, working, it]
    )
    intercept = value[-1, 1, working, it] - value[-1, 0, working, it] * slope
    res[res == np.max(value[:, 1, working, it])] = (
        intercept + slope * x[res == np.max(value[:, 1, working, it])]
    )

    return res


# Calculation of probability to choose work, if a worker today
def chpr(x, lambda_):
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
