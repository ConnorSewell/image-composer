import cv2
import HistogramHandler
import numpy as np
from skimage import data, img_as_float
from skimage.measure import structural_similarity as ssim



def getAverageColor(image, index, bins):
    (h, w, _) = image.shape
    histogram = cv2.calcHist([image], [index], None, [bins], [0, bins])
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(image_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    x = 0
    for i in range(0, len(histogram)):
        x += (int(histogram[i]) * i)
    return x / (w * h)


def getEdgeImage(img):
    edges = cv2.resize(img, (25, 25), interpolation=cv2.INTER_AREA)
    edges = cv2.fastNlMeansDenoisingColored(edges, None, 10, 10, 7, 21)
    edges = cv2.Canny(edges, 180, 200)
    return edges.flatten()


# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html For Hough Transform. Accessed: 14/11/2016
def getHoughTransformLines(img):
    length = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    if lines is not None:
        length = len(lines)

    return length


def calc_entropy(histogram):
    return 0

#http://scikit-image.org/docs/dev/auto_examples/plot_ssim.html


def calc_ssim(img):
    imgSSIM = ssim(img, img, dynamic_range=img.max() - img.min())
    return imgSSIM


#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def harris_corner_detection(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    return img.flatten()


def siftDescriptor(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(img, None)


def extractFeature(image):
    entry = {}
    entry["b"] = getAverageColor(image, 0, 256)
    entry["g"] = getAverageColor(image, 1, 256)
    entry["r"] = getAverageColor(image, 2, 256)
    entry["histogram"] = HistogramHandler.calc_histogram(image)
    return entry

