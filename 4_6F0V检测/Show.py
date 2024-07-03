import matplotlib.pyplot as plt



def display_images_with_titles(image_dict):
    num_images = len(image_dict)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, image_dict.items()):
        ax.set_title(title)
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    plt.show()