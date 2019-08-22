import numpy as np
import scipy.interpolate as scin

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


def budget(it, savings, shocks, working, ngridm, n_quad_points, r, coeffs_age_poly):
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
        (ngridm, n_quad_points), income(it + 1, shocks, coeffs_age_poly) * working
    ).T + np.full((n_quad_points, ngridm), savings * (1 + r))

    return w1


def mbudget(ngridm, n_quad_points, r):
    """Marginal budget:
    Derivative of budget with respect to savings"""

    mw1 = np.full((n_quad_points, ngridm), (1 + r))

    return mw1


# Value function for worker
# interpolate and extrapolate are potentially substitutable by the interpolate function below


def value_function(working, it, x, value, beta, theta, duw):
    x = x.flatten("F")

    res = np.full(x.shape, np.nan)

    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    mask = x < value[1, 0, working, it]

    # Calculate t+1 value function in the constrained region
    res[mask] = util(x[mask], working, theta, duw) + beta * value[0, 1, working, it]

    # Calculate t+1 value function in non-constrained region
    # inter- and extrapolate
    interpolation = scin.interp1d(
        value[:, 0, working, it],
        value[:, 1, working, it],
        bounds_error=False,
        fill_value="extrapolate",
    )
    res[~mask] = interpolation(x[~mask])

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


def egm_step(
    value,
    policy,
    savingsgrid,
    quadstnorm,
    period,
    Tbar,
    ngridm,
    cfloor,
    n_quad_points,
    r,
    coeffs_age_poly,
    theta,
    duw,
    beta,
    lambda_,
    sigma,
    quadw,
):
    """This function executes the EGM step of the algorithm."""

    for choice in [0, 1]:
        wk1 = budget(
            period,
            savingsgrid,
            quadstnorm * sigma,
            choice,
            ngridm,
            n_quad_points,
            r,
            coeffs_age_poly,
        )

        wk1[wk1 < cfloor] = cfloor

        # Value function
        vl1 = np.full((2, ngridm * n_quad_points), np.nan)

        if period + 1 == Tbar - 1:
            vl1[0, :] = util(wk1, 0, theta, duw).flatten("F")
            vl1[1, :] = util(wk1, 1, theta, duw).flatten("F")
        else:
            vl1[1, :] = value_function(
                1, period + 1, wk1, value, beta, theta, duw
            )  # value function in t+1 if choice in t+1 is work
            vl1[0, :] = value_function(0, period + 1, wk1, value, beta, theta, duw)

            # Probability of choosing work in t+1
        if choice == 0:
            # Probability of choosing work in t+1
            pr1 = np.full(2500, 0.00)
        else:
            pr1 = chpr(vl1, lambda_)

        # Next period consumption based on interpolation and extrapolation
        # given grid points and associated consumption
        cons10_interpolate = scin.interp1d(
            policy[:, 0, 0, period + 1],
            policy[:, 1, 0, period + 1],
            bounds_error=False,
            fill_value="extrapolate",
        )
        cons10_flat = cons10_interpolate(wk1).flatten("F")

        cons11_interpolate = scin.interp1d(
            policy[:, 0, 1, period + 1],
            policy[:, 1, 1, period + 1],
            bounds_error=False,
            fill_value="extrapolate",
        )
        # extrapolate linearly right of max grid point
        cons11_flat = cons11_interpolate(wk1).flatten("F")

        # Marginal utility of expected consumption next period
        mu1 = pr1 * mutil(cons11_flat, theta) + (1 - pr1) * mutil(cons10_flat, theta)

        # Marginal budget
        # Note: Constant for this model formulation (1+r)
        mwk1 = mbudget(ngridm, n_quad_points, r)

        # RHS of Euler eq., p 337, integrate out error of y
        rhs = np.dot(quadw.T, np.multiply(mu1.reshape(wk1.shape, order="F"), mwk1))
        # Current period consumption from Euler equation

        cons0 = imutil(beta * rhs, theta)
        # Update containers related to consumption
        policy[1:, 1, choice, period] = cons0
        policy[1:, 0, choice, period] = savingsgrid + cons0

        if choice == 1:
            # Calculate continuation value
            ev = np.dot(quadw.T, logsum(vl1, lambda_).reshape(wk1.shape, order="F"))
        else:
            ev = np.dot(quadw.T, vl1[0, :].reshape(wk1.shape, order="F"))
        # Update value function related containers
        value[1:, 1, choice, period] = util(cons0, choice, theta, duw) + beta * ev
        value[1:, 0, choice, period] = savingsgrid + cons0
        value[0, 1, choice, period] = ev[0]
    return value
