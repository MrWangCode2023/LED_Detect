import numpy as np


def calculate_lnu(L, Lcenter):
    """
    Calculate the brightness uniformity (LNU) for a single point.

    :param L: Brightness of the test point.
    :param Lcenter: Brightness of the center test point.
    :return: Brightness uniformity (LNU) as a percentage.
    """
    if Lcenter == 0:
        return np.inf
    return (L / Lcenter) * 100