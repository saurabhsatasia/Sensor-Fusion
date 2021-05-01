import numpy as np
from math import sin, cos, sqrt


def cartesian_to_polar(x, y, vx, vy, THRESH = 0.0001):
    """
    Converts 2D cartesian position and velocity coordinates to polar coordinates.
    :param x, y: Position components in cartesian coordinates
    :param vx, vy: Velocity compnents in cartesian coordinates
    :return:
        rho, d_rho: floats - radius and velocity magnitude respectively
        phi: float - angle in radians
    """
    rho = sqrt(x**2 + y**2)
    phi = np.arctan(y, x)

    if rho < THRESH:
        print("WARNING: in cartesian_to_polar(): d_squared < THRESH")
        rho, phi, d_rho = 0, 0, 0
    else:
        d_rho = (x*vx + y*vy) / rho
    return rho, phi, d_rho


def polar_to_cartesian(rho, phi, d_rho):
    """
    Converts 2D polar coordinates into cartesian coordinates
    :Input params:
        rho: float - radius magnitude
        d_rho: float - velocity magnitude
        phi: float - angle in radians
    :return:
        x, y: floats - position components in cartesian
        vx, vy: floats - velocity components in cartesian
    """
    x, y = rho * cos(phi), rho * sin(phi)
    vx, vy = cos(phi), d_rho * sin(phi)
    return x, y, rho, d_rho


def time_difference(t1, t2):
    """
    Computes the time difference in microseconds(float) of two epoch times values in seconds(int)
    :input params:
        t1: int - previous epoch time in seconds
        t2: int - current epoch time in seconds
    :returns: float - the time difference in seconds
    """
    return (t2-t1) / 1000000.0


def get_RMSE(predictions, truths):
    """
    Compute the RMSE of the attributes of 2 lists of DataPoint() instances
    :input params:
        predictions - a list of DataPoint() instances
        truths - a list of DataPoint() instances
    :returns:
        px, py, vx, vy - floats: The RMSE of each respective DataPoint() attributes
    """
    pxs, pys, vxs, vys = [], [], [], []
    for preds, truths in zip(predictions, truths):
        ppx, ppy, pvx, pvy = preds.get()
        tpx, tpy, tvx, tvy = truths.get()

        pxs += [(ppx - tpx)**2]
        pys += [(ppy - tpy)**2]
        vxs += [(pvx - tvx)**2]
        vys += [(pvy - tvy)**2]

    px, py = sqrt(np.mean(pxs)), sqrt(np.mean(pys))
    vx, vy = sqrt(np.mean(vxs)), sqrt(np.mean(vys))
    return px, py, vx, vy


def calculate_jacobian(px, py, vx, vy, THRESH=0.0001, ZERO_REPLACEMENT=0.0001):
    """
      Calculates the Jacobian given for four state variables
      Args:
        px, py, vx, vy : floats - four state variables in the system
        THRESH - minimum value of squared distance to return a non-zero matrix
        ZERO_REPLACEMENT - value to replace zero to avoid division by zero error
      Returns:
        H : the jacobian matrix expressed as a 4 x 4 numpy matrix with float values
    """

    d_squared = px * px + py * py
    d = sqrt(d_squared)
    d_cubed = d_squared * d

    if d_squared < THRESH:

        print("WARNING: in calculate_jacobian(): d_squared < THRESH")
        H = np.matrix(np.zeros([3, 4]))

    else:

        r11 = px / d
        r12 = py / d
        r21 = -py / d_squared
        r22 = px / d_squared
        r31 = py * (vx * py - vy * px) / d_cubed
        r32 = px * (vy * px - vx * py) / d_cubed

        H = np.matrix([[r11, r12, 0, 0],
                       [r21, r22, 0, 0],
                       [r31, r32, r11, r12]])

    return H
