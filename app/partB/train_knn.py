import numpy as np
from matplotlib import pyplot as plt
import cv2
import graphlab
from app.utils.utils import get_numpy_data, get_residual_sum_of_squares

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html
#^ Accessed 15/11/2016 at 00:39 - Adapted code for below.

train_images = graphlab.SFrame.read_csv('hough_line_count_only_train.csv')
graphlab.cross_validation.shuffle(train_images)

features = train_images.column_names()
features.remove('class')

X,Y = get_numpy_data(train_images, features, 'class')

responses = np.array(Y, dtype='f')
trainData = np.array(X, dtype='f')

test_images = graphlab.SFrame.read_csv('hough_line_count_only_test.csv')
testSetFeatures = test_images.column_names()
testSetFeatures.remove('class')

X_test, Y_test = get_numpy_data(test_images, testSetFeatures, 'class')

testData = np.array(X_test, dtype='f')
testResponses = np.array(Y_test, dtype='f')

knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

ret, results, neighbours, dist = knn.findNearest(testData, 10)

print len(results)

man_made = 0
natural = 0
matches = 0

for i in xrange(0, 500):
    if results[i] == 0:
        man_made = man_made + 1
    else:
        natural = natural + 1
    if results[i] == testResponses[i]:
        matches = matches + 1


print man_made
print natural
print matches

#print "result: ", results, "\n"
#print "neighbours: ", neighbours, "\n"
#print "distance: ", dist

plt.show()