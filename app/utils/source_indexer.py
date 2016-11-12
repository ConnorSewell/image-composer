import glob
import json
import cv2
import ntpath
import os

import feature_extractor


DB_PATH = "/home/ouanixi/Work/image-composer/dataset/images/"
INDEX_PATH = "/home/ouanixi/Work/image-composer/dataset/index/"


def convertImage(path, length):
    image = cv2.imread(path)
    image = cv2.GaussianBlur(image,(21,21),0)
    image = cv2.resize(image, (length, length), interpolation=cv2.INTER_AREA)
    cv2.imwrite(INDEX_PATH + ntpath.basename(path), image)
    return image


def getFileList():
    included_extenstions = ['jpg', 'bmp', 'png', 'gif'];
    return [fn for fn in os.listdir(DB_PATH) if any([fn.endswith(ext) for ext in included_extenstions])];


def start_indexing(length):
    index = []
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)
    files = glob.glob(DB_PATH + "/" + "*.jpg")

    entry = {}

    for file in files:
        print "Processing file: " + file
        image = convertImage(file, length)
        entry = feature_extractor.extractFeature(image)
        entry["file"] = ntpath.basename(file)
        index.append(entry)

    with open(INDEX_PATH + "histogram.index", 'w') as outfile:
        json.dump(index, outfile, indent=4)

    print ("Index written to: " + INDEX_PATH + "histogram.index")
