import numpy as np
from copy import deepcopy
from ret import util


def create_container(num_grid, num_periods, max_asset, theta, cost_work):
    # We are currently facing a problem since our result arrays do not have the same
    # length. Instead of a multidimensional array we should use a list in which we
    # store the different arrays.
    # What would this mean for our functions?
    m_grid = np.linspace(0, max_asset, num_grid)
    # Set up list containers
    policy = [
        [np.full((num_grid + 1, 2), np.nan) for k in range(2)]
        for i in range(num_periods)
    ]
    value = [
        [np.full((num_grid + 1, 2), np.nan) for k in range(2)]
        for i in range(num_periods)
    ]
    # Handling of last period and first elements
    # policy
    for k in range(num_periods):
        for choice in range(2):
            value[k][choice][0, 0] = 0.00
            policy[k][choice][0, :] = 0.00

    policy[num_periods - 1][0][1:, 0] = deepcopy(m_grid)
    policy[num_periods - 1][1][1:, 0] = deepcopy(m_grid)
    policy[num_periods - 1][0][1:, 1] = deepcopy(policy[num_periods - 1][0][1:, 0])
    policy[num_periods - 1][1][1:, 1] = deepcopy(policy[num_periods - 1][1][1:, 0])
    # value
    value[num_periods - 1][0][2:, 0] = util(
        policy[num_periods - 1][0][2:, 0], 0, theta, cost_work
    )
    value[num_periods - 1][0][2:, 1] = util(
        policy[num_periods - 1][1][2:, 0], 1, theta, cost_work
    )
    value[num_periods - 1][1][2:, 0] = util(
        policy[num_periods - 1][0][2:, 0], 0, theta, cost_work
    )
    value[num_periods - 1][1][2:, 1] = util(
        policy[num_periods - 1][1][2:, 0], 1, theta, cost_work
    )

    value[num_periods - 1][0][0:2] = 0.00
    value[num_periods - 1][1][0:2] = 0.00

    # The time and the choice dimension are now extracted from our array format.
    # Instead we are using a list for saving the results which allows us to alter the
    # legth of the arrays
    return policy, value
