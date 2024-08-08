import numpy as np

def t6_resoluton_calculate(flare_width, dmp=2, pmr=5):
    resolution = flare_width / (2 * dmp * pmr)

    return resolution