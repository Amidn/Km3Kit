import numpy as np

def unit_vector(vector, **kwargs):
    """Returns the unit vector of the vector."""
    vector = np.array(vector)
    out_shape = vector.shape
    vector = np.atleast_2d(vector)
    unit = vector / np.linalg.norm(vector, axis=1, **kwargs)[:, None]
    return unit.reshape(out_shape)

def angle_between(v1, v2, axis=0):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    if axis == 0:
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.inner(v1_u, v2_u), -1, 1))
    elif axis == 1:
        raise ValueError("axis=1 not used in this context.")
    else:
        raise ValueError("unsupported axis")

def zenith(v):
    """Return the zenith angle in radians.

    Parameters
    ----------
    v : array (x, y, z)

    Notes
    -----
    Defined as 'Angle respective to downgoing'.
    Downgoing event: zenith = 0
    Horizont: 90deg
    Upgoing: zenith = 180deg
    """
    return angle_between((0, 0, -1), v)