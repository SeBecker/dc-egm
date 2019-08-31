"""This module contains the solution process"""
import numpy as np
import scipy.stats as scps
from scipy.special.orthogonal import ps_roots

from dcegm.solve.solve_auxiliary import create_container
from dcegm.solve.solve_auxiliary import egm_step
from dcegm.solve.solve_auxiliary import secondary_envelope_wrapper


def solve_retirmenet_model(
    ngridm,
    n_quad_points,
    r,
    coeffs_age_poly,
    theta,
    duw,
    beta,
    lambda_,
    sigma,
    mmax,
    Tbar,
    cfloor=0.001,
):
    # Initialize grids
    quadstnorm = scps.norm.ppf(ps_roots(n_quad_points)[0])
    quadw = ps_roots(n_quad_points)[1]

    # define savingsgrid
    savingsgrid = np.linspace(0, mmax, ngridm)

    # Set up list containers
    policy, value = create_container()
    #

    for period in range(Tbar, -1, -1):
        for choice in [1, 0]:
            value, policy, ev = egm_step(
                value,
                policy,
                choice,
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
            )
            if choice == 1:
                value_, policy_ = secondary_envelope_wrapper(
                    value, policy, period, theta, duw, beta, ev, ngridm
                )
                value[period][choice] = value_
                policy[period][choice] = policy_

    return value, policy
