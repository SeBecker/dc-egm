"""This module contains the simulation method."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from dcegm.retirement.ret import choice_probabilities
from dcegm.retirement.ret import income
from dcegm.retirement.ret import value_function


def simulate(
    value,
    policy,
    num_periods,
    cost_work,
    theta,
    beta,
    lambda_,
    sigma,
    r,
    coeffs_age_poly,
    init=[10, 30],
    num_sims=10,
    seed=7134,
):

    # Set seed
    np.random.seed(seed)

    # Create containers
    wealth0 = np.full((num_sims, num_periods), np.nan)
    wealth1 = np.full((num_sims, num_periods), np.nan)
    cons = np.full((num_sims, num_periods), np.nan)
    shock = np.full((num_sims, num_periods), np.nan)
    income_ = np.full((num_sims, num_periods), np.nan)
    worker = np.full((num_sims, num_periods), np.nan)
    prob_work = np.full((num_sims, num_periods), np.nan)
    ret_age = np.full((num_sims, 1), np.nan)
    vl1 = np.full((2, num_sims), np.nan)

    # Set initial t=0 values

    # Draw inperiodial wealth
    wealth0[:, 0] = init[0] + np.random.uniform(0, 1, num_sims) * (init[1] - init[0])

    # Set status of all individuals (given by nsims) to working, i.e. 1
    worker[:, 0] = 1

    # Fill in containers

    # Next period value function
    vl1[0, :] = value_function(
        0, 0, wealth0[:, 0], value, beta, theta, cost_work
    )  # retirement
    vl1[1, :] = value_function(
        1, 0, wealth0[:, 0], value, beta, theta, cost_work
    )  # work

    # Choice probabilperiody of working
    prob_work[:, 0] = choice_probabilities(vl1, lambda_)

    working = (prob_work[:, 0] > np.random.uniform(0, 1, num_sims)).astype(int)

    cons[:, 0][working == 0], cons[:, 0][working == 1] = cons_t0(
        wealth0, policy, working
    )

    wealth1[:, 0] = wealth0[:, 0] - cons[:, 0]

    # Record current period choice
    for period in range(1, num_periods - 1):
        worker[:, period] = working
        # Shock, here no shock since set sigma = 0 for m0
        shock[:, period][worker[:, period] == 1] = (
            norm.ppf(np.random.uniform(0, 1, sum(working))) * sigma
        )

        # Fill in retirement age
        ret_age[(worker[:, period - 1] == 1) & (worker[:, period] == 0)] = period

        wealth0, wealth1, cons, prob_work, working, shock = sim_periods(
            wealth0,
            wealth1,
            cons,
            worker,
            num_sims,
            prob_work,
            shock,
            period,
            coeffs_age_poly,
            lambda_,
            income_,
            value,
            policy,
            beta,
            theta,
            cost_work,
            r,
        )
    df = create_dataframe(
        wealth0, wealth1, cons, worker, income_, shock, ret_age, num_sims, num_periods
    )

    return df


def cons_t0(wealth0, policy, working):
    """This function calculates the cons in period 0"""
    cons10 = np.interp(wealth0[:, 0], policy[1][0].T[0], policy[1][0].T[1])
    # extrapolate linearly right of max grid point
    slope = (policy[1][0].T[1][-2] - policy[1][0].T[1][-1]) / (
        policy[1][0].T[0][-2] - policy[+1][0].T[0][-1]
    )
    intercept = policy[1][0].T[1][-1] - policy[1][0].T[0][-1] * slope
    cons10[cons10 == np.max(policy[1][0].T[1])] = (
        intercept + slope * wealth0[:, 0][cons10 == np.max(policy[1][0].T[1])]
    )
    cons10_flat = cons10.flatten("F")

    cons11 = np.interp(wealth0[:, 0], policy[1][1].T[0], policy[1][1].T[1])
    # extrapolate linearly right of max grid point
    slope = (policy[1][1].T[1][-2] - policy[1][1].T[1][-1]) / (
        policy[1][1].T[0][-2] - policy[1][1].T[0][-1]
    )
    intercept = policy[1][1].T[1][-1] - policy[1][1].T[0][-1] * slope
    cons11[cons11 == np.max(policy[1][1].T[1])] = (
        intercept + slope * wealth0[:, 0][cons11 == np.max(policy[1][1].T[1])]
    )
    cons11_flat = cons11.flatten("F")

    return cons10_flat[working == 0], cons11_flat[working == 1]


def sim_periods(
    wealth0,
    wealth1,
    cons,
    worker,
    num_sims,
    prob_work,
    shock,
    period,
    coeffs_age_poly,
    lambda_,
    income_,
    value,
    policy,
    beta,
    theta,
    cost_work,
    r,
):

    # Income
    income_[:, period] = 0
    income_[:, period][worker[:, period] == 1] = income(
        period, shock[:, period], coeffs_age_poly
    )[worker[:, period] == 1]

    # M_t+1
    # MatLab code should be equvalent to calculating correct income for workers and retired
    # and just adding savings times interest
    # No extra need for further differentiating between retired and working
    wealth0[:, period] = income_[:, period] + wealth1[:, period - 1] * (1 + r)

    # Next period value function
    vl1 = np.full((2, num_sims), np.nan)

    vl1[0, :] = value_function(
        0, period, wealth0[:, period], value, beta, theta, cost_work
    )  # retirement
    vl1[1, :] = value_function(
        1, period, wealth0[:, period], value, beta, theta, cost_work
    )  # work

    # Choice probabilperiody of working
    prob_work[:, period] = choice_probabilities(vl1, lambda_)

    # Record current period choice
    working = (prob_work[:, period] > np.random.uniform(0, 1, num_sims)).astype(int)
    # retirement is absorbing state
    working[worker[:, period] == 0] = 0.0

    # Calculate current period cons

    cons10 = np.interp(
        wealth0[:, period], policy[period + 1][0].T[0], policy[period + 1][0].T[1]
    )
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][0].T[1][-2] - policy[period + 1][0].T[1][-1]) / (
        policy[period + 1][0].T[0][-2] - policy[period + 1][0].T[0][-1]
    )
    intercept = policy[period + 1][0].T[1][-1] - policy[period + 1][0].T[0][-1] * slope
    cons10[cons10 == np.max(policy[period + 1][0].T[1])] = (
        intercept
        + slope * wealth0[:, period][cons10 == np.max(policy[period + 1][0].T[1])]
    )
    cons10_flat = cons10.flatten("F")

    cons11 = np.interp(
        wealth0[:, period], policy[period + 1][1].T[0], policy[period + 1][1].T[1]
    )
    # extrapolate linearly right of max grid point
    slope = (policy[period + 1][1].T[1][-2] - policy[period + 1][1].T[1][-1]) / (
        policy[period + 1][1].T[0][-2] - policy[period + 1][1].T[0][-1]
    )
    intercept = policy[period + 1][1].T[1][-1] - policy[period + 1][1].T[0][-1] * slope
    cons11[cons11 == np.max(policy[period + 1][1].T[1])] = (
        intercept
        + slope * wealth0[:, period][cons11 == np.max(policy[period + 1][1].T[1])]
    )
    cons11_flat = cons11.flatten("F")

    cons[:, period][working == 1] = cons11_flat[working == 1]
    cons[:, period][working == 0] = cons10_flat[working == 0]

    wealth1[:, period] = wealth0[:, period] - cons[:, period]

    return wealth0, wealth1, cons, prob_work, working, shock


def create_dataframe(
    wealth0, wealth1, cons, worker, income_, shock, ret_age, num_sims, num_periods
):
    """This function processes the results so that they are composed in a pandas
    dataframe object.
    """

    # Set up multiindex object
    index = pd.MultiIndex.from_product(
        [np.arange(num_sims), np.arange(num_periods)], names=["identifier", "period"]
    )

    # Define column names object
    columns = [
        "wealth0",
        "wealth1",
        "consumption",
        "working",
        "income",
        "retirement_age",
        "shock",
    ]

    # Process data
    data = np.vstack(
        [
            wealth0.flatten("C"),
            wealth1.flatten("C"),
            cons.flatten("C"),
            worker.flatten("C"),
            income_.flatten("C"),
            ret_age.repeat(num_periods).flatten("C"),
            shock.flatten("C"),
        ]
    )

    df = pd.DataFrame(data.T, index, columns)

    return df
