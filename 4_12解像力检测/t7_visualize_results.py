import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

import cv2
import numpy as np
import matplotlib.pyplot as plt


def t7_visualize_results(show_image, peak_x, reso_x, flare_x, smoothed_esf):
    """
    Visualizes the image with marked peak, resolution, and flare lines,
    and plots the smoothed edge spread function (ESF).

    Parameters:
        image (numpy.ndarray): The input image to be visualized.
        peak_x (int): The x-coordinate of the peak.
        reso_x (list of int): The x-coordinates for resolution lines.
        flare_x (list of int): The x-coordinates for flare lines.
        smoothed_esf (numpy.ndarray): The smoothed edge spread function values.
    """
    # Mark peak line
    cv2.line(show_image, (peak_x, 0), (peak_x, show_image.shape[0]), (0, 0, 255), 1)

    # Mark resolution lines
    for x in reso_x:
        if x is not None:  # Ensure x is not None
            cv2.line(show_image, (x, 0), (x, show_image.shape[0]), (255, 0, 0), 1)

    # Mark flare lines
    for x in flare_x:
        if x is not None:  # Ensure x is not None
            cv2.line(show_image, (x, 0), (x, show_image.shape[0]), (0, 255, 0), 1)

    # Create plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image show")
    plt.imshow(cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("ESF")
    plt.plot(smoothed_esf, color="black", label="Luminance")
    plt.axvline(x=peak_x, color="red", linestyle="--", label="Peak")  # Mark peak
    for x in reso_x:
        if x is not None:  # Ensure x is not None
            plt.axvline(x=x, color="blue", linestyle="--", label="Resolution")  # Mark resolution
    for x in flare_x:
        if x is not None:  # Ensure x is not None
            plt.axvline(x=x, color="green", linestyle="--", label="Flare")  # Mark flare
    plt.xlabel("X-axis coordinates")
    plt.ylabel("Luminance")
    plt.legend()

    plt.tight_layout()
    plt.show()




