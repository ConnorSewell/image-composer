import numpy as np
import cv2
from matplotlib import pyplot as plt
import graphlab
from app.utils.utils import get_numpy_data, get_residual_sum_of_squares

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html Accessed 17/11/2016 @ 00:23

#Read descriptors
trainDescriptors = 0
trainClassifiers = 0
testDescriptors = 0

#Generated test classifiers
testClasses = 0
#Classifiers for tests from file
knownTestClassifiers = 0

bf = cv2.BFMatcher()

loopTopMatches = 0
bestMatches = 0
bestMatchIndex = 0
classifiers = []

#good = [[]]
for i in xrange(0, 500):
    for k in xrange(0, 1000):
        matches = bf.knnMatch(testDescriptors[i], trainDescriptors[k], k = 2)
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                loopTopMatches += 1
        if loopTopMatches > bestMatches:
            bestMatches = loopTopMatches
            bestMatchIndex = k
            loopTopMatches = 0

    classifiers[i] = trainClassifiers[bestMatchIndex]

matches = 0

for i in xrange(0, 500):
    if classifiers[i] == knownTestClassifiers[i]:
        matches += 1

print matches