import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


def preprocess(imgOriginal):
    img_grayscale = extractValue(imgOriginal)
    img_max_contrast_grayscale = maximizeContrast(img_grayscale)
    height, width = img_grayscale.shape
    img_blurred = np.zeros((height, width, 1), np.uint8)
    img_blurred = cv2.GaussianBlur(img_max_contrast_grayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(img_blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return img_grayscale, imgThresh


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    img_hsv = np.zeros((height, width, 3), np.uint8)
    img_hsv = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, img_value = cv2.split(img_hsv)

    return img_value


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    img_top_hat = np.zeros((height, width, 1), np.uint8)
    img_black_hat = np.zeros((height, width, 1), np.uint8)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_top_hat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuring_element)
    img_black_hat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuring_element)

    img_grayscale_plus_top_hat = cv2.add(imgGrayscale, img_top_hat)
    img_grayscale_plus_top_hat_minus_black_hat = cv2.subtract(img_grayscale_plus_top_hat, img_black_hat)

    return img_grayscale_plus_top_hat_minus_black_hat
