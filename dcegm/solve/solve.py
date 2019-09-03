"""This module contains the solution process"""
import numpy as np
import scipy.stats as scps
from scipy.special.orthogonal import ps_roots

from dcegm.solve.egm_step import egm_step
from dcegm.solve.secondary_envelope import secondary_envelope_wrapper
from dcegm.solve.solve_auxiliary import create_container


def solve_retirement_model(
    num_grid,
    n_quad_points,
    r,
    coeffs_age_poly,
    theta,
    cost_work,
    beta,
    lambda_,
    sigma,
    mmax,
    num_periods,
    cfloor=0.001,
):
    # Initialize grids
    quadstnorm = scps.norm.ppf(ps_roots(n_quad_points)[0])
    quadw = ps_roots(n_quad_points)[1]

    # define savingsgrid
    savingsgrid = np.linspace(0, mmax, num_grid)

    # Set up list containers
    policy, value = create_container(
        num_grid, num_periods, savingsgrid, theta, cost_work
    )
    # state = 0 retirement
    # state = 1 worker
    for period in range(num_periods - 2, -1, -1):
        # TODO: For state = 0, no need to solve egm.
        for state in [1, 0]:
            value, policy, ev = egm_step(
                value,
                policy,
                state,
                savingsgrid,
                quadstnorm,
                period,
                num_periods,
                num_grid,
                cfloor,
                n_quad_points,
                r,
                coeffs_age_poly,
                theta,
                cost_work,
                beta,
                lambda_,
                sigma,
                quadw,
            )
            if state == 1:
                value_, policy_ = secondary_envelope_wrapper(
                    value, policy, period, theta, cost_work, beta, ev, num_grid
                )
                value[period][state] = value_
                policy[period][state] = policy_

    return value, policy
