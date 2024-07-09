import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_data(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)


def esf(pixel_values):
    esf1 = pixel_values
    esf = smooth_data(esf1)
    return esf


def lsf(esf):
    lsf1 = np.diff(esf, axis=0)
    lsf = smooth_data(lsf1)
    return lsf


def mtf(lsf):
    mtf = np.abs(np.fft.fft(lsf, axis=0))
    mtf1 = mtf / np.max(mtf)
    mtf = smooth_data(mtf1)
    return mtf
