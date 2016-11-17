import cv2
from app.utils import feature_extractor
from app.utils.utils import shuffle_lines
import numpy as np
import csv
import os
from app.utils import HistogramHandler


#DATAPATH = "/home/ouanixi/Work/image-composer/dataset/raw"
DATAPATH = "C:/Users/Connor/Desktop/NewImages/Training"
MAN_TRAINING = "manmade_training.txt"
NAT_TRAINING = "natural_training.txt"

#sift = cv2.SIFT()


def load_files(fname):
    image_path = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            image_path.append(DATAPATH + "/" + line.split('.', 1)[1].rstrip('\n'))
    return image_path


def create_training_set():
    # get manmade images
    man_image_names = load_files(DATAPATH + '/' + MAN_TRAINING)
    nat_image_names = load_files(DATAPATH + '/' + NAT_TRAINING)
    feature_list = []

    for name in man_image_names:
        print name
        img = cv2.imread(name)
        #edges = feature_extractor.getEdgeImage(img)
        img_dict = {}
        histogram = HistogramHandler.calc_histogram(img)
        pix = 'pix_ '
        img_dict[pix] = histogram
        img_dict["class"] = 0
        feature_list.append(img_dict)

    for name in nat_image_names:
        img = cv2.imread(name)
        img_dict = {}
        histogram = HistogramHandler.calc_histogram(img)
        pix = 'pix_ '
        img_dict[pix] = histogram
        img_dict["class"] = 1
        feature_list.append(img_dict)

    return feature_list


def make_training_csv():
    toCSV = create_training_set()
    keys = toCSV[0].keys()
    with open('histogram_only_train.csv', 'ab') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)

make_training_csv()