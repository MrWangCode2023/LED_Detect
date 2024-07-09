def ESF(line_image):
    img = line_image.copy()
    line_data = pix_img(img)
    # plot line_data
    ESF_f = f1(line_data)
    LSF_f = ESF_to_LSF(ESF_f)
    MTF_f = LSF_to_MTF(LSF_f)
    return MTF_f