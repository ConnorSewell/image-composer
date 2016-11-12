import cv2
import numpy as np


#http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

def calc_correlation(histogram1, histogram2):
    correlation = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CORREL)
    return correlation


def calc_bhattacharyya_dist(histogram1, histogram2):
    distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_BHATTACHARYYA)
    return distance


def calc_chi_squared_dist(histogram1, histogram2):
    distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CHISQR)
    return distance


#http://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def calc_euclidean_dist(histogram1, histogram2):
    distance = np.sqrt(np.sum((histogram1 - histogram2)**2))
    return distance


def calc_hist_intersection(histogram1, histogram2):
    intersection = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_INTERSECT)
    return intersection


def calc_histogram(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py
    hist = cv2.calcHist(image_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    #http://stackoverflow.com/questions/9390592/drawing-histogram-in-opencv-python
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


#ttp://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.htmlh
#def sobel_comparison(image1, image2):
#    image1_x_sobel = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=5)
#    image1_y_sobel = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=5)
#    image2_x_sobel = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=5)
#    image2_y_sobel = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=5)
#    return 0


#def compare_entropy(histogram1, histogram2):
#    return 0


#def calculate_entropy(histogram, bin_count):
#    for i in range(0, bin_count):
#        print histogram[i]

#    positive_entropy = histogram
#    entropy = -positive_entropy
#    return entropy


#https://www.mathworks.com/help/stats/knnsearch.html
#^ For Part B

#http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images