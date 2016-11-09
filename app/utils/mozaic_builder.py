import json
import cv2
import math
import sys
import feature_extractor

#INDEX_PATH = "/home/ouanixi/Work/image-composer/dataset/index/"
INDEX_PATH = "C:\Users\Connor\Desktop\image-composer\dataset\index"

def readIndex():
    json_data = open(INDEX_PATH + "histogram.index").read()
    return json.loads(json_data)


def preparInputImage(path, tileSize):
    i = cv2.imread(path)
    (h, w, _) = i.shape
    i = cv2.resize(i, (w / tileSize * tileSize, h / tileSize * tileSize))
    return i


def preparePatch(path, tileSize):
    image = cv2.imread(INDEX_PATH + path)
    image = cv2.resize(image, (tileSize, tileSize))
    return image


def calcDistance(fts1, fts2, vectors):
    distance = 0
    for vec in vectors:
        distance += math.pow(fts1[vec] - fts2[vec], 2)
    return math.sqrt(distance)


def getIndexImage(fts, index, vectors):
    minDistance = sys.maxint
    imagefile = ""
    for item in index:
        distance = calcDistance(fts, item, vectors)
        if distance < minDistance:
            minDistance = distance
            imagefile = item["file"]
    return imagefile


def processLine(i, w, index, inputImage, tileSize, channels):
    for j in range(0, w / tileSize):
        roi = inputImage[i * tileSize:(i + 1) * tileSize, j * tileSize:(j + 1) * tileSize]
        fts = feature_extractor.extractFeature(roi)
        patch = preparePatch(getIndexImage(fts, index, channels), tileSize)
        inputImage[i * tileSize:(i + 1) * tileSize, j * tileSize:(j + 1) * tileSize] = patch
        cv2.imshow("Progress", inputImage)
        cv2.waitKey(1)


def main(inputImagePath, tileSize):

    channels = ['r','g', 'b']

    # read index + input image
    index = readIndex()
    inputImage = preparInputImage(inputImagePath, tileSize)

    (h, w, _) = inputImage.shape

    inputImage = cv2.resize(inputImage, (w / tileSize * tileSize, h / tileSize * tileSize))
    print inputImage.shape

    for i in range(0, h / tileSize):
        processLine(i, w, index, inputImage, tileSize, channels)

    print "Finished processing of image"

    cv2.imwrite("Result2.jpg", inputImage)
