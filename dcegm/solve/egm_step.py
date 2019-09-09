import numpy as np

from dcegm.retirement.ret import budget
from dcegm.retirement.ret import choice_probabilities
from dcegm.retirement.ret import imutil
from dcegm.retirement.ret import logsum
from dcegm.retirement.ret import mbudget
from dcegm.retirement.ret import mutil
from dcegm.retirement.ret import util
from dcegm.retirement.ret import value_function


def egm_step(
    value,
    policy,
    choice,
    savingsgrid,
    quadstnorm,
    period,
    num_periods,
    num_grid,
    cons_floor,
    n_quad_points,
    interest,
    coeffs_age_poly,
    theta,
    duw,
    beta,
    lambda_,
    sigma,
    quadw,
):
    """This function executes the EGM step of the algorithm."""
    wealth_t1 = budget(
        period,
        savingsgrid,
        quadstnorm * sigma,
        choice,
        num_grid,
        n_quad_points,
        interest,
        coeffs_age_poly,
    )
    wealth_t1[wealth_t1 < cons_floor] = cons_floor  # Replace with retirement saftey net
    # TODO: Extract calculation of value function
    # Value function
    value_t1 = np.full((2, num_grid * n_quad_points), np.nan)
    if period + 1 == num_periods - 1:
        value_t1[0, :] = util(wealth_t1, 0, theta, duw).flatten("F")
        value_t1[1, :] = util(wealth_t1, 1, theta, duw).flatten("F")
    else:
        value_t1[1, :] = value_function(
            1, period + 1, wealth_t1, value, beta, theta, duw
        )  # value function in t+1 if choice in t+1 is work
        value_t1[0, :] = value_function(
            0, period + 1, wealth_t1, value, beta, theta, duw
        )

    # TODO: Extract calculation of probabilities
    # Probability of choosing work in t+1
    if choice == 0:
        # Probability of choosing work in t+1
        choice_prob_t1 = np.full(n_quad_points * num_grid, 0.00)
    else:
        choice_prob_t1 = choice_probabilities(value_t1, lambda_)

    # TODO: Extract consumption and produce one array with one dimension for choice
    # Next period consumption based on interpolation and extrapolation
    # given grid points and associated consumption
    cons10 = np.interp(
        wealth_t1, policy[period + 1][0].T[0], policy[period + 1][0].T[1]
    )
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][0].T[1][-2] - policy[period + 1][0].T[1][-1]) / (
        policy[period + 1][0].T[0][-2] - policy[period + 1][0].T[0][-1]
    )

    intercept = policy[period + 1][0].T[1][-1] - policy[period + 1][0].T[0][-1] * slope
    cons10[cons10 == np.max(policy[period + 1][0].T[1])] = (
        intercept + slope * wealth_t1[cons10 == np.max(policy[period + 1][0].T[1])]
    )
    cons10_flat = cons10.flatten("F")

    cons11 = np.interp(
        wealth_t1, policy[period + 1][1].T[0], policy[period + 1][1].T[1]
    )
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][1].T[1][-2] - policy[period + 1][1].T[1][-1]) / (
        policy[period + 1][1].T[0][-2] - policy[period + 1][1].T[0][-1]
    )
    intercept = policy[period + 1][1].T[1][-1] - policy[period + 1][1].T[0][-1] * slope
    cons11[cons11 == np.max(policy[period + 1][1].T[1])] = (
        intercept + slope * wealth_t1[cons11 == np.max(policy[period + 1][1].T[1])]
    )
    cons11_flat = cons11.flatten("F")
    # TODO: Extract function for marginal utility
    # Marginal utility of expected consumption next period
    marg_ut_t1 = choice_prob_t1 * mutil(cons11_flat, theta) + (
        1 - choice_prob_t1
    ) * mutil(cons10_flat, theta)
    # Marginal budget
    # Note: Constant for this model formulation (1+r)
    marg_bud_t1 = mbudget(num_grid, n_quad_points, interest)
    # RHS of Euler eq., p 337, integrate out error of y
    rhs_eul = np.dot(
        quadw.T,
        np.multiply(marg_ut_t1.reshape(wealth_t1.shape, order="F"), marg_bud_t1),
    )
    # Current period consumption from Euler equation
    cons_t0 = imutil(beta * rhs_eul, theta)
    # Update containers related to consumption
    policy[period][choice][1:, 1] = cons_t0
    policy[period][choice][1:, 0] = savingsgrid + cons_t0

    if choice == 1:
        # Calculate continuation value
        ev = np.dot(
            quadw.T, logsum(value_t1, lambda_).reshape(wealth_t1.shape, order="F")
        )
    else:
        ev = np.dot(quadw.T, value_t1[0, :].reshape(wealth_t1.shape, order="F"))

    value[period][choice][1:, 1] = util(cons_t0, choice, theta, duw) + beta * ev
    value[period][choice][1:, 0] = savingsgrid + cons_t0
    value[period][choice][0, 1] = ev[0]

    # Why is ev returned without?
    return value, policy, ev
