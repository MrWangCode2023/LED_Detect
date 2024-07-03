import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_result_image(original, show_image, luminance, illuminance):
    # Normalize the illuminance image to the range 0-255
    norm_illuminance = cv2.normalize(illuminance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_illuminance = np.uint8(norm_illuminance)

    # Apply a color map to the normalized illuminance image
    color_mapped_illuminance = cv2.applyColorMap(norm_illuminance, cv2.COLORMAP_JET)

    # Convert images to RGB (from BGR)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    show_image_rgb = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
    luminance_rgb = cv2.cvtColor(luminance, cv2.COLOR_GRAY2RGB)
    color_mapped_illuminance_rgb = cv2.cvtColor(color_mapped_illuminance, cv2.COLOR_BGR2RGB)

    # Create a figure to display the images
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Emax Detection Results', fontsize=16)

    # Display images
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(show_image_rgb)
    axes[0, 1].set_title('Emax Point')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(luminance_rgb, cmap='gray')
    axes[1, 0].set_title('Luminance Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(color_mapped_illuminance_rgb)
    axes[1, 1].set_title('Illuminance Image (Color Mapped)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()