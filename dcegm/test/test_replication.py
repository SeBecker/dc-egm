import pickle

import numpy as np

from dcegm.dcegm_config import TEST_RESOURCES_DIR
from dcegm.solve.solve import solve_retirement_model


def test1():

    # Load results from the matlab code
    m0_value = pickle.load(open(TEST_RESOURCES_DIR + "/m0_value_new.pkl", "rb"))
    m0_policy = pickle.load(open(TEST_RESOURCES_DIR + "/m0_policy_new.pkl", "rb"))

    # Number of periods (fist period is t=1)
    num_periods = 25

    # Number of grid points over assets
    num_grid = 500

    # Maximum level of assets
    mmax = 50

    # Number of quadrature points used in calculation of expectations
    n_quad_points = 5

    # Interest rate
    r = 0.05

    # Discount factor
    beta = 0.95

    # Standard deviation of log-normally distributed income shocks
    sigma = 0.00

    # Disutility of work
    cost_work = 0.35

    # CRRA coefficient (log utility if ==1)
    theta = 1.95

    coeffs_age_poly = np.array([0.75, 0.04, -0.0002])

    # Scale of the EV taste shocks
    lambda_ = 2.2204e-16

    value, policy = solve_retirement_model(
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
    )

    for period in range(24):
        for choice in [1, 0]:
            np.testing.assert_array_almost_equal(
                value[period][choice], m0_value[period][choice]
            )
            np.testing.assert_array_almost_equal(
                policy[period][choice], m0_policy[period][choice]
            )


def test2():

    # Load results from the matlab code
    m5_value = pickle.load(open(TEST_RESOURCES_DIR + "/m5_value_new.pkl", "rb"))
    m5_policy = pickle.load(open(TEST_RESOURCES_DIR + "/m5_policy_new.pkl", "rb"))

    # Number of periods (fist period is t=1)
    num_periods = 25

    # Number of grid points over assets
    num_grid = 500

    # Maximum level of assets
    mmax = 50

    # Number of quadrature points used in calculation of expectations
    n_quad_points = 5

    # Interest rate
    r = 0.05

    # Discount factor
    beta = 1/(1+r)

    # Standard deviation of log-normally distributed income shocks
    sigma = 0.35

    # Disutility of work
    cost_work = 0.35

    # CRRA coefficient (log utility if ==1)
    theta = 1.95

    coeffs_age_poly = np.array([0.75, 0.04, -0.0002])

    # Scale of the EV taste shocks
    lambda_ = 0.2

    value, policy = solve_retirement_model(
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
    )

    for period in range(24):
        for choice in [1, 0]:
            np.testing.assert_array_almost_equal(
                value[period][choice], m5_value[period][choice]
            )
            np.testing.assert_array_almost_equal(
                policy[period][choice], m5_policy[period][choice]
            )
