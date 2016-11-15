import cv2
from app.utils import feature_extractor
from app.utils.utils import shuffle_lines
import numpy as np
import csv

DATAPATH = "/home/ouanixi/Work/image-composer/dataset/raw"
MAN_TRAINING = "manmade_test.txt"
NAT_TRAINING = "natural_test.txt"


def load_files(fname):
    image_path = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            image_path.append(DATAPATH + line.split('.',1)[1].rstrip('\n'))
    return image_path


def create_training_set():
    # get manmade images
    man_image_names = load_files(DATAPATH + '/' + MAN_TRAINING)
    nat_image_names = load_files(DATAPATH + '/' + NAT_TRAINING)
    feature_list = []
    for name in man_image_names:
        img = cv2.imread(name)
        edges = feature_extractor.getEdgeImage(img)
        img_dict = {}
        for i in xrange(len(edges)):
            pix = 'pix_' + str(i)
            img_dict[pix] = edges[i]
        img_dict["class"] = 0
        feature_list.append(img_dict)

    for name in nat_image_names:
        img = cv2.imread(name)
        edges = feature_extractor.getEdgeImage(img)
        for i in xrange(len(edges)):
            pix = 'pix_' + str(i)
            img_dict[pix] = edges[i]
        img_dict["class"] = 1
        feature_list.append(img_dict)

    return feature_list


def make_training_csv():
    toCSV = create_training_set()
    keys = toCSV[0].keys()
    with open('edges_only_test.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)


make_training_csv()