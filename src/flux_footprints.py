def footprint_area(xr, yr):
    """
    Compute area enclosed by a footprint contour.

    Parameters
    ----------
    xr : array-like
        x-coordinates of the contour (meters).
    yr : array-like
        y-coordinates of the contour (meters).

    Returns
    -------
    float
        Area in square meters.
    """
    x = np.asarray(xr, dtype=float)
    y = np.asarray(yr, dtype=float)
    if x.size != y.size:
        raise ValueError("xr and yr must have the same length")
    if x.size < 3:
        raise ValueError("At least 3 points are required to form a polygon")

    # Ensure the polygon is closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Classic shoelace (Gauss) area formula
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area