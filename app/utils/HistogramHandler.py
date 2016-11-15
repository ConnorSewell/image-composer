import cv2

from scipy.spatial import distance as dist


def calc_distance(histogram1, histogram2, method=None):
    if method == cv2.HISTCMP_CHISQR:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CHISQR)
    elif method == cv2.HISTCMP_CHISQR_ALT:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CHISQR_ALT)
    elif method == cv2.HISTCMP_BHATTACHARYYA:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_BHATTACHARYYA)
    elif method == cv2.HISTCMP_CORREL:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CORREL)
    elif method == cv2.HISTCMP_HELLINGER:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_HELLINGER)
    elif method == cv2.HISTCMP_INTERSECT:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_INTERSECT)
    elif method == cv2.HISTCMP_KL_DIV:
        distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_KL_DIV)
    else:
        distance = dist.euclidean(histogram1, histogram2)
    return distance


def calc_histogram(image):
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # #https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py
    # hist = cv2.calcHist(image_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    # #http://stackoverflow.com/questions/9390592/drawing-histogram-in-opencv-python
    # cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


    # https://www.mathworks.com/help/stats/knnsearch.html
    # ^ For Part B
