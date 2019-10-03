import numpy as np

from dcegm.retirement.ret import budget
from dcegm.retirement.ret import choice_probs_worker
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
    cost_work,
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
    # Value function
    value_t1 = next_period_value(
        value, wealth_t1, cost_work, period, num_periods, theta, beta
    )

    # Marginal utility of expected consumption next period
    marg_ut_t1 = next_period_marg_ut(
        choice, value_t1, wealth_t1, lambda_, policy, period, theta
    )

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

    ev = expected_value(choice, value_t1, quadw, lambda_, wealth_t1)

    value[period][choice][1:, 1] = util(cons_t0, choice, theta, cost_work) + beta * ev
    value[period][choice][1:, 0] = savingsgrid + cons_t0
    value[period][choice][0, 1] = ev[0]

    # Why is ev returned without beta?
    return value, policy, ev


def next_period_value(value, wealth, cost_work, period, num_periods, theta, beta):
    value_t1 = np.full((2, wealth.shape[0] * wealth.shape[1]), np.nan)
    if period + 1 == num_periods - 1:
        value_t1[0, :] = util(wealth, 0, theta, cost_work).flatten("F")
        value_t1[1, :] = util(wealth, 1, theta, cost_work).flatten("F")
    else:
        value_t1[1, :] = value_function(
            1, period + 1, wealth, value, beta, theta, cost_work
        )  # value function in t+1 if choice in t+1 is work
        value_t1[0, :] = value_function(
            0, period + 1, wealth, value, beta, theta, cost_work
        )
    return value_t1


def next_period_choice_probs(choice, value, lambda_, n_quad_points, num_grid):
    if choice == 0:
        # Probability of choosing work in t+1
        choice_prob = np.full(n_quad_points * num_grid, 0.00)
    else:
        choice_prob = choice_probs_worker(value, lambda_)
    return choice_prob


def next_period_consumption(policy, period, wealth):
    cons = np.empty((2, wealth.shape[0] * wealth.shape[1]))
    cons_0 = np.interp(wealth, policy[period + 1][0].T[0], policy[period + 1][0].T[1])
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][0].T[1][-2] - policy[period + 1][0].T[1][-1]) / (
        policy[period + 1][0].T[0][-2] - policy[period + 1][0].T[0][-1]
    )

    intercept = policy[period + 1][0].T[1][-1] - policy[period + 1][0].T[0][-1] * slope
    cons_0[cons_0 == np.max(policy[period + 1][0].T[1])] = (
        intercept + slope * wealth[cons_0 == np.max(policy[period + 1][0].T[1])]
    )
    cons[0, :] = cons_0.flatten("F")

    cons_1 = np.interp(wealth, policy[period + 1][1].T[0], policy[period + 1][1].T[1])
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][1].T[1][-2] - policy[period + 1][1].T[1][-1]) / (
        policy[period + 1][1].T[0][-2] - policy[period + 1][1].T[0][-1]
    )
    intercept = policy[period + 1][1].T[1][-1] - policy[period + 1][1].T[0][-1] * slope
    cons_1[cons_1 == np.max(policy[period + 1][1].T[1])] = (
        intercept + slope * wealth[cons_1 == np.max(policy[period + 1][1].T[1])]
    )
    cons[1, :] = cons_1.flatten("F")
    return cons


def next_period_marg_ut(choice, value_t1, wealth_t1, lambda_, policy, period, theta):
    num_grid, n_quad_points = wealth_t1.shape
    # Probability of choosing work in t+1
    choice_prob_t1 = next_period_choice_probs(
        choice, value_t1, lambda_, n_quad_points, num_grid
    )

    # Next period consumption based on interpolation and extrapolation
    # given grid points and associated consumption
    cons_t1 = next_period_consumption(policy, period, wealth_t1)
    return choice_prob_t1 * mutil(cons_t1[1, :], theta) + (1 - choice_prob_t1) * mutil(
        cons_t1[0, :], theta
    )


def expected_value(choice, value_t1, quadw, lambda_, wealth_t1):
    if choice == 1:
        # Calculate continuation value
        return np.dot(
            quadw.T, logsum(value_t1, lambda_).reshape(wealth_t1.shape, order="F")
        )
    else:
        return np.dot(quadw.T, value_t1[0, :].reshape(wealth_t1.shape, order="F"))
