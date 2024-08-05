import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, image):
        self.image = image

    def BGR2GRAY(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def GaussianBlur(self, kernel_size=(5, 5)):
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return self.image

    def Edges(self, low_threshold=50, high_threshold=150):
        return cv2.Canny(self.image, low_threshold, high_threshold)

    def EqualizeHist(self):
        if len(self.image.shape) == 2:
            self.image = cv2.equalizeHist(self.image)
        else:
            ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            self.image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return self.image

    def CLAHE(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(self.image.shape) == 2:
            self.image = clahe.apply(self.image)
        else:
            ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            self.image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return self.image

    def Contrast_Stretching(self):
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype("uint8")
        self.image = cv2.LUT(self.image, table)
        return self.image

    def Gamma_correction(self, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)
        return self.image

    def Dilation(self, kernel_size=(5, 5), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=iterations)
        return self.image

    def Erosion(self, kernel_size=(5, 5), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=iterations)
        return self.image

    def display_image(self, window_name="Image"):
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
