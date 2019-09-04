"""This module contains some auxiliary methods for the solving process. We should
split this in several modules."""
from copy import deepcopy

import numpy as np
import scipy.interpolate as scin

from dcegm.retirement.ret import util


def create_container(num_grid, num_periods, savingsgrid, theta, cost_work):
    # We are currently facing a problem since our result arrays do not have the same
    # length. Instead of a multidimensional array we should use a list in which we
    # store the different arrays.
    # What would this mean for our functions?
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

    policy[num_periods - 1][0][1:, 0] = deepcopy(savingsgrid)
    policy[num_periods - 1][1][1:, 0] = deepcopy(savingsgrid)
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


def diff(obj, pl2, significance=5):
    x1 = np.round(pl2[0] * (10 ** significance)) * (10 ** (-significance))
    y1 = np.round(pl2[1] * (10 ** significance)) * (10 ** (-significance))
    x = np.round(obj[0] * (10 ** significance)) * (10 ** (-significance))
    y = np.round(obj[1] * (10 ** significance)) * (10 ** (-significance))
    indx = list(
        set(
            [np.where(x == i)[0][0] for i in np.setdiff1d(x, x1)]
            + [np.where(y == i)[0][0] for i in np.setdiff1d(y, y1)]
        )
    )

    return np.array(indx)


def aux_function(x, obj1, obj2):
    x = [x]
    value, extr = np.subtract(interpolate(x, obj1), interpolate(x, obj2))
    return value


def interpolate(xx, obj, one=False):
    """Interpolation function"""
    if not one:
        interpolation = scin.interp1d(
            obj[0], obj[1], bounds_error=False, fill_value="extrapolate"
        )
        container = interpolation(xx)
        extrapolate = [
            True if (i > max(obj[0])) | (i < min(obj[0])) else False for i in xx
        ]
    else:
        container = []
        extrapolate = []

        for poly in obj:
            interpolation = scin.interp1d(
                poly[0], poly[1], bounds_error=False, fill_value="extrapolate"
            )
            container += [interpolation(xx)]
            extrapolate += [
                np.array(
                    [
                        True if (i > max(poly[0])) | (i < min(poly[0])) else False
                        for i in xx
                    ]
                )
            ]
    return container, extrapolate


def chop(obj, j, repeat=None):
    """This function separates the grid into 1,..,j and j+1,...N parts."""
    for k in range(1):
        if j > len(obj[k]):
            j = len(obj[k])
        part1 = np.stack([obj[0][: j + 1], obj[1][: j + 1]])

        if repeat is not None:
            # If repeat == True the boundary points are included in both arrays
            if repeat:
                part2 = np.stack([obj[0][j:], obj[1][j:]])
            else:
                part2 = np.stack([obj[0][j + 1 :], obj[1][j + 1 :]])
        if repeat is None:
            part2 = np.array([])

    return part1, part2
