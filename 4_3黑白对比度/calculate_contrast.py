import numpy as np


def calculate_contrast(Lwhite, Ldark):
    """
    Calculate contrast given the brightness values for white and black images.

    :param Lwhite: List of brightness values for the white image.
    :param Ldark: List of brightness values for the black image.
    :return: List of contrast values.
    """
    contrast_values = []
    for lw, ld in zip(Lwhite, Ldark):
        if ld == 0:
            contrast_values.append(np.inf)  # Avoid division by zero
        else:
            contrast_values.append((lw / ld) * 100)
    return contrast_values