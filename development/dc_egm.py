from copy import deepcopy

import numpy as np
import scipy.interpolate as scin
from numpy.matlib import repmat
from scipy.optimize import brenth


def polyline(x, y=None, label=None):
    """This function converts the """
    # create output container
    obj = np.empty((2, len(x)))

    # condition output on the number on function inputs
    if x is not None:
        x = np.array(x).flatten()
        obj[0, :] = x
    if y is not None:
        y = np.array(y).flatten()
        obj[1, :] = y

    return obj


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
