import torch


def tupleset(t, i, value):
    """
    Modify a specific index in a tuple and return a new tuple.

    Args:
        :t (tuple): The input tuple.
        :i (int): The index to modify.
        :value (any): The new value to insert at index `i`.

    Returns:
        :ell (tuple): A new tuple with the specified index modified.
    """
    ell = list(t)
    ell[i] = value
    return tuple(ell)


def _basic_simpson(y, start, stop, x, dx, axis):
    """
    Core implementation of Simpson's rule along a given axis.
    If x is None, assumes equally spaced data with spacing dx.

    Parameters
    ----------
    Args:
        :y (torch.Tensor): Function values to be integrated.
        :start (int): Starting index along the axis.
        :stop (int): Stopping index along the axis.
        :x (torch.Tensor, optional): Points at which y is sampled.
        :dx (float): Spacing between points if x is not provided.
        :axis (int): Axis along which to integrate.

    Returns:
        :result (torch.Tensor): Estimated integral using Simpson's rule.
    """
    nd = len(y.shape)

    if start is None:
        start = 0

    step = 2
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
    slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = torch.sum(y[slice0] + 4.0 * y[slice1] + y[slice2], dim=axis)
        result *= dx / 3.0
    else:
        # Account for different spacings
        h = torch.diff(x, dim=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        h0 = h[sl0].float()
        h1 = h[sl1].float()
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = torch.where(h1 != 0, h0 / h1, torch.zeros_like(h0))

        tmp = hsum / 6.0 * (
            y[slice0] * (2.0 - torch.where(h0divh1 != 0, 1.0 / h0divh1, torch.zeros_like(h0divh1))) +
            y[slice1] * (hsum * torch.where(hprod != 0, hsum / hprod, torch.zeros_like(hsum))) +
            y[slice2] * (2.0 - h0divh1)
        )
        result = torch.sum(tmp, dim=axis)

    return result


def simpson(y, *, x=None, dx=1.0, axis=-1):
    """
    Integrate `y` along the given axis using Simpson's rule.

    Args:
        :y (torch.Tensor): Tensor to be integrated.
        :x (torch.Tensor, optional): Points at which `y` is sampled. If not given, uniform spacing `dx` is assumed.
        :dx (float, optional): Spacing between points if `x` is not provided. Default is 1.0.
        :axis (int, optional): Axis along which to integrate. Default is the last axis.

    Returns:
        :result (torch.Tensor): The estimated integral using Simpson's rule.

    Notes
    -----
    If the number of points along `axis` is even, Simpson's rule requires an adjustment
    since it only works with an odd number of points. The function handles this case by
    applying a correction to the last interval.
    """

    if len(y) < 2:
        raise ValueError("y must have at least 2 elements to integrate.")

    y = torch.as_tensor(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    returnshape = 0

    if x is not None:
        x = torch.as_tensor(x)
        if len(x.shape) == 1:
            # Reshape `x` to broadcast along the specified axis
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.view(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError(
                "If given, x must be 1-D or have the same shape as y.")

        if x.shape[axis] != N:
            raise ValueError(
                "If given, x must have the same length as y along the specified axis.")

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = torch.diff(x, dim=axis).to(torch.float64)
                h = [diffs[hm2].squeeze(dim=axis),
                     diffs[hm1].squeeze(dim=axis)]
            else:
                h = torch.tensor([dx, dx], dtype=torch.float64)

            # Correction for the last interval according to Cartwright
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = torch.where(den != 0, num / den, torch.zeros_like(den))

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = torch.where(den != 0, num / den, torch.zeros_like(den))

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = torch.where(den != 0, num / den, torch.zeros_like(den))

            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.view(saveshape)

    return result


def hypergeometric_series(a, b, c, z, max_iter=10000, tolerance=1.0e-6):
    """
    Compute the hypergeometric series _2F_1(a, b; c; z) using a series expansion.

    Args:
        :a (float): First parameter of the hypergeometric function.
        :b (float): Second parameter of the hypergeometric function.
        :c (float): Third parameter of the hypergeometric function.
        :z :(torch.Tensor): Argument of the hypergeometric function.
        :max_iter (int): Maximum number of iterations for convergence.
        :tolerance (float): Tolerance for convergence.

    Returns:
        F (torch.Tensor): The computed hypergeometric series value.
    """

    if torch.any(torch.abs(z) >= 1.0):
        raise ValueError(
            "This implementation may not be accurate for |z| >= 1.")

    # Initialize variables
    F = torch.ones_like(z)
    term = torch.ones_like(z)
    ap = a
    bp = b
    cp = c

    for i in range(1, max_iter + 1):
        # Update term and F
        term = term * ap * bp * z / (cp * i)
        F = F + term

        # Update parameters
        ap += 1.0
        bp += 1.0
        cp += 1.0

        # Check convergence
        if torch.all(torch.abs(term) < torch.abs(F) * tolerance):
            break

    return F


def hyp2f1(a, b, c, z):
    """
    Compute the hypergeometric function _2F_1(a, b; c; z).

    Args:
        :a (float): First parameter of the hypergeometric function.
        :b (float): Second parameter of the hypergeometric function.
        :c (float): Third parameter of the hypergeometric function.
        :z :(torch.Tensor): Argument of the hypergeometric function.

    Returns:
        :result (torch.Tensor): The computed hypergeometric function value.
    """

    # Initialize result tensor
    result = torch.zeros_like(z)

    # Handle cases where |z| < 1
    m = torch.abs(z) < 1.0
    if torch.any(m):
        result[m] = hypergeometric_series(a, b, c, z[m])

    # Handle cases where z < -1 using Euler's transformation
    m = z < -1.0
    if torch.any(m):
        new_z = z[m] / (z[m] - 1.0)
        result[m] = (1.0 - z[m]) ** (-a) * \
            hypergeometric_series(a, c - b, c, new_z)

    # Handle cases where z >= 1
    if torch.any(z >= 1.0):
        result[z >= 1.0] = torch.tensor(float("inf"))

    return result
