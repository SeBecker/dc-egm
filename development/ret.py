from copy import deepcopy

import numpy as np
import scipy.interpolate as scin
from numpy.matlib import repmat
from scipy.optimize import brenth

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


def budget_old(it, savings, shocks, working, ngridm, expn, r, coeffs_age_poly):
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
        (ngridm, expn), income(it, shocks, coeffs_age_poly) * working
    ).T + np.full((expn, ngridm), savings * (1 + r))

    return w1


def mbudget(ngridm, n_quad_points, r):
    """Marginal budget:
    Derivative of budget with respect to savings"""

    mw1 = np.full((n_quad_points, ngridm), (1 + r))

    return mw1


# Value function for worker
# interpolate and extrapolate are potentially substitutable by the interpolate function
# below


def value_function(working, it, x, value, beta, theta, duw):
    x = x.flatten("F")

    res = np.full(x.shape, np.nan)
    # Mark constrained region
    # credit constraint between 1st (M_{t+1) = 0) and second point (A_{t+1} = 0)
    mask = x < value[it][working][1, 0]

    # Calculate t+1 value function in the constrained region
    res[mask] = util(x[mask], working, theta, duw) + beta * value[it][working][0, 1]

    # Calculate t+1 value function in non-constrained region
    # inter- and extrapolate
    interpolation = scin.interp1d(
        value[it][working][:, 0],
        value[it][working][:, 1],
        bounds_error=False,
        fill_value="extrapolate",
        kind="linear",
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
        cons10 = np.interp(wk1, policy[period + 1][0].T[0], policy[period + 1][0].T[1])
        # extrapolate linearly right of max grid point
        slope = (policy[period + 1][0].T[1][-2] - policy[period + 1][0].T[1][-1]) / (
            policy[period + 1][0].T[0][-2] - policy[period + 1][0].T[0][-1]
        )
        intercept = (
            policy[period + 1][0].T[1][-1] - policy[period + 1][0].T[0][-1] * slope
        )
        cons10[cons10 == np.max(policy[period + 1][0].T[1])] = (
            intercept + slope * wk1[cons10 == np.max(policy[period + 1][0].T[1])]
        )
        cons10_flat = cons10.flatten("F")

        cons11 = np.interp(wk1, policy[period + 1][1].T[0], policy[period + 1][1].T[1])
        # extrapolate linearly right of max grid point
        slope = (policy[period + 1][1].T[1][-2] - policy[period + 1][1].T[1][-1]) / (
            policy[period + 1][1].T[0][-2] - policy[period + 1][1].T[0][-1]
        )
        intercept = (
            policy[period + 1][1].T[1][-1] - policy[period + 1][1].T[0][-1] * slope
        )
        cons11[cons11 == np.max(policy[period + 1][1].T[1])] = (
            intercept + slope * wk1[cons11 == np.max(policy[period + 1][1].T[1])]
        )
        cons11_flat = cons11.flatten("F")
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
        policy[period][choice][1:, 1] = cons0
        policy[period][choice][1:, 0] = savingsgrid + cons0

        if choice == 1:
            # Calculate continuation value
            ev = np.dot(quadw.T, logsum(vl1, lambda_).reshape(wk1.shape, order="F"))
        else:
            ev = np.dot(quadw.T, vl1[0, :].reshape(wk1.shape, order="F"))
        value[period][choice][1:, 1] = util(cons0, choice, theta, duw) + beta * ev
        value[period][choice][1:, 0] = savingsgrid + cons0
        value[period][choice][0, 1] = ev[0]

        if choice == 1:
            value_, policy_ = secondary_envelope_wrapper(
                value, policy, period, theta, duw, beta, ev, ngridm
            )
            value[period][choice] = value_
            policy[period][choice] = policy_

    return value, policy


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


# Upper envelope changed compared to original
# Most importantly, assigning the proper output values to result_upper was completely missing
def upper_envelope(obj, fullinterval=False, intersection=False):

    # Assert if input of polyline objects is not an array (length==1)
    assert len(obj) != 1, "Upper envelope is meant for an array of polylines"
    length = []

    # copy original input
    aux_object = deepcopy(obj)
    # check length of polyline entries and drop polylines with x-length == 0
    for k1 in range(len(obj)):
        length += [(len(obj[k1][0]), len(obj[k1][1]))]
        aux_object[k1] = [i for i in aux_object[k1] if len(i) != 0]

    # Get all unique values of x
    xx = np.array([])
    for k1 in range(len(aux_object)):
        xx = np.append(xx, aux_object[k1][0].astype(list))
    xx = np.array([i for i in np.unique(xx)])
    # set up containers
    interpolated = np.empty((len(obj), len(xx)))
    extrapolated = np.empty((len(obj), len(xx)))

    # interpolate for each unique value of x
    for counter in range(len(obj)):
        inter, extra = interpolate(xx, obj[counter])
        interpolated[counter, :] = inter
        extrapolated[counter, :] = extra
    extrapolated = extrapolated.astype(bool)
    if not fullinterval:
        mask = np.sum(extrapolated, axis=0) > 0
        container = np.empty((interpolated.shape[0], int(mask.sum())))
        for i in range(interpolated.shape[0]):
            container[i, :] = np.extract(mask, interpolated[i, :])
        interpolated = container
        xx = xx[mask]
        n = sum(~mask)
    else:
        interpolated[extrapolated] = -np.inf
        n = len(xx)

    # create upper envelope
    maxinterpolated = repmat(interpolated.max(axis=0), m=interpolated.shape[0], n=1)
    top = interpolated == maxinterpolated
    top = top.astype(int)
    # Initialise container
    # result_upper = polyline(xx, maxinterpolated[1,:]) ## This does not seem to work as
    #  intended
    result_inter = np.empty((2, 0))
    container1 = np.array([])
    container2 = np.array([])

    # Containers for collection of valid polyline points
    result_upper_cont_x = [xx[0]]  # Added line
    result_upper_cont_y = [interpolated[0, 0]]  # Added line

    while_operator = True
    while while_operator:
        k0 = np.where(top[:, 0] == 1)[0][0]
        for i in range(1, n):
            k1 = np.where(top[:, i] == 1)[0][0]
            if k1 != k0:
                ln1 = k0
                ln2 = k1
                xx1 = xx[i - 1]
                xx2 = xx[i]
                y1, extr1 = interpolate([xx1, xx2], aux_object[ln1])
                y2, extr2 = interpolate([xx1, xx2], aux_object[ln2])
                if np.all(~np.stack([extr1, extr2])) & np.all(abs(y1 - y2) > 0):
                    xx3 = brenth(
                        aux_function, xx1, xx2, args=(aux_object[ln1], aux_object[ln2])
                    )
                    xx3f, _ = interpolate([xx3], aux_object[ln1])
                    # set up containers
                    interpolated2 = np.empty((len(obj), 1))
                    extrapolated2 = np.empty((len(obj), 1))

                    # interpolate for each unique value of x
                    for counter in range(len(obj)):
                        inter2, extra2 = interpolate([xx3], obj[counter])
                        interpolated2[counter] = inter2
                        extrapolated2[counter] = extra2

                    extrapolated2 = extrapolated2.astype(bool)
                    interpolated2[extrapolated2] = -np.inf
                    maxinterpolated2 = repmat(interpolated2.max(), m=len(obj), n=1)
                    ln3 = np.where(interpolated2 == maxinterpolated2)[0][0]
                    if (ln3 == ln1) | (ln3 == ln2):

                        # there are no other functions above!
                        # add the intersection point
                        result_upper_cont_x.append(xx3)
                        result_upper_cont_y.append(float(xx3f))

                        if intersection:
                            container1 = np.append(container1, [xx3])
                            container2 = np.append(container2, [xx3f])

                        if ln2 == k1:

                            while_operator = False

                        else:
                            ln1 = ln2
                            xx1 = xx3
                            ln2 = k1
                            xx2 = xx[i]
                    else:
                        ln2 = ln3
                        xx2 = xx3

            # This was missing before!!!! : replicates MatLab lines 342-346 which are
            # extremely important
            # Add point to container if it is on the currently highest line
            if any(abs(obj[k1][0] - xx[i]) < 2.2204e-16) is True:
                result_upper_cont_x.append(xx[i])
                result_upper_cont_y.append(maxinterpolated[0, i])

            k0 = k1

        # Collect results
        result_inter = np.empty((2, len(container1)))
        result_inter[0], result_inter[1] = container1, container2
        result_upper = [
            np.array(result_upper_cont_x),
            np.array(result_upper_cont_y),
        ]  # Added line

    return result_upper, result_inter


def secondary_envelope(obj):
    result = []
    newdots = []
    index_removed = []

    sect = []
    cur = deepcopy(obj)
    # Find discontinutiy
    ii = cur[0][1:] > cur[0][:-1]
    # Substitute for matlab while true loop
    i = 1
    while_operator = True
    while while_operator:
        j = np.where([ii[counter] != ii[0] for counter in range(len(ii))])[0]
        if len(j) == 0:
            if i > 1:
                sect += [cur]
            while_operator = False
        else:
            j = min(j)

            sect_container, cur = chop(cur, j, True)
            sect += [sect_container]
            ii = ii[j:]
            i += 1
    # yes we can use np.sort instead of the pre-specified function from the upper
    # envelope notebook
    if len(sect) > 1:
        sect = [np.sort(i) for i in sect]
        result_container, newdots_container = upper_envelope(sect, True, True)
        index_removed_container = diff(obj, result_container, 10)
    else:
        result_container = obj
        index_removed_container = np.array([])
        newdots_container = np.stack([np.array([]), np.array([])])

    result += [result_container]
    newdots += [newdots_container]
    index_removed += [index_removed_container]

    return np.array(result[0]), newdots[0], index_removed[0]


def secondary_envelope_wrapper(value, policy, period, theta, duw, beta, ev, ngridm):
    # get minimal x value
    minx = min(value[period][1].T[0][1:])
    if value[period][1].T[0][1] <= minx:
        value_, newdots, del_index = secondary_envelope(value[period][1].T)
    else:
        x1 = np.linspace(minx, value[period][1].T[0][1], np.round(ngridm / 10))
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
