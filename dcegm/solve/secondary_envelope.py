from copy import deepcopy

import numpy as np
import scipy.interpolate as scin

from dcegm.retirement.ret import util
from dcegm.solve.solve_auxiliary import chop
from dcegm.solve.solve_auxiliary import diff
from dcegm.solve.upper_envelope import upper_envelope


def secondary_envelope_wrapper(value, policy, period, theta, duw, beta, ev, num_grid):
    # get minimal x value
    minx = min(value[period][1].T[0][1:])
    # Why not equality?
    if value[period][1].T[0][1] <= minx:
        value_, newdots, del_index = secondary_envelope(value[period][1].T)
    else:
        x1 = np.linspace(minx, value[period][1].T[0][1], int(np.round(num_grid / 10)))
        x1 = x1[:-1]
        y1 = util(x1, 1.0, theta, duw) + beta * ev[0]
        value_x = np.append(x1, value[period][1].T[0][1:])
        value_y = np.append(y1, value[period][1].T[1][1:])
        value_aux = np.stack([value_x, value_y])
        policy_x = np.append(x1, policy[period][1].T[0][1:])
        policy_y = np.append(x1, policy[period][1].T[1][1:])
        policy[period][1] = deepcopy(np.stack([policy_x, policy_y]).T)
        value_, newdots, del_index = secondary_envelope(value_aux)
        aux_array = np.zeros((2, 1))
        aux_array[1] = ev[0]
        value_ = np.hstack([aux_array, value_])
    if len(del_index) > 0:
        new_policy = []

        for counter, i in enumerate(newdots[0]):
            j = max(
                [
                    i
                    for i in np.where(policy[period][1].T[0] < newdots[0][counter])[0]
                    if i not in del_index
                ]
            )
            interpolation1 = scin.interp1d(
                policy[period][1].T[0][j : j + 2],
                policy[period][1].T[1][j : j + 2],
                bounds_error=False,
                fill_value="extrapolate",
            )
            point1 = interpolation1(newdots[0][counter])
            j = min(
                [
                    i
                    for i in np.where(policy[period][1].T[0] > newdots[0][counter])[0]
                    if i not in del_index
                ]
            )

            interpolation2 = scin.interp1d(
                policy[period][1].T[0][j - 1 : j + 1],
                policy[period][1].T[1][j - 1 : j + 1],
                bounds_error=False,
                fill_value="extrapolate",
            )
            point2 = interpolation2(newdots[0][counter])
            new_policy += [np.array([newdots[0][counter], point1, point2])]
        policy_x = np.array(
            [i for c, i in enumerate(policy[period][1].T[0]) if c not in del_index]
        )
        policy_y = np.array(
            [i for c, i in enumerate(policy[period][1].T[1]) if c not in del_index]
        )
        for k in range(len(new_policy)):
            j = [i for i in np.where(policy_x > new_policy[k][0])[0]][0]
            policy_x = np.insert(policy_x, j, new_policy[k][0])
            policy_x = np.insert(policy_x, j + 1, new_policy[k][0] - 0.001 * 2.2204e-16)
            policy_y = np.insert(policy_y, j, new_policy[k][1])
            policy_y = np.insert(policy_y, j + 1, new_policy[k][2])
        policy_ = np.stack([policy_x, policy_y]).T

    else:
        policy_ = policy[period][1]
    if policy_[0][0] != 0.0:
        aux_array = np.zeros((1, 2))
        policy_ = np.vstack([aux_array, policy_])

    return value_.T, policy_


def secondary_envelope(values):
    result = []
    newdots = []
    index_removed = []

    section = []
    cur = deepcopy(values)
    # Find discontinutiy, i.e. the following value smaller then the predecessor
    discont_points = cur[0][1:] > cur[0][:-1]
    # Substitute for matlab while true loop
    i = 1
    while_operator = True
    # TODO: Get more pythonic loop
    # Algorithm 3, line 3+4
    while while_operator:
        j = np.where(
            [
                discont_points[counter] != discont_points[0]
                for counter in range(len(discont_points))
            ]
        )[0]
        if len(j) == 0:
            if i > 1:
                section += [cur]
            while_operator = False
        else:
            j = min(j)

            sect_container, cur = chop(cur, j, True)
            section += [sect_container]
            discont_points = discont_points[j:]
            i += 1
    # yes we can use np.sort instead of the pre-specified function from the upper
    # envelope notebook
    # If we have more than one section, apply upper envelope
    if len(section) > 1:
        section = [np.sort(i) for i in section]
        result_container, newdots_container = upper_envelope(section, True, True)
        index_removed_container = diff(values, result_container, 10)
    else:
        result_container = values
        index_removed_container = np.array([])
        newdots_container = np.stack([np.array([]), np.array([])])

    result += [result_container]
    newdots += [newdots_container]
    index_removed += [index_removed_container]

    return np.array(result[0]), newdots[0], index_removed[0]
