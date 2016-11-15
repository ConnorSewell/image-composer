
from sklearn.neural_network import MLPClassifier
from app.utils.utils import get_numpy_data, get_residual_sum_of_squares
from sklearn import svm
import graphlab


train_images = graphlab.SFrame.read_csv('edges_only_train.csv')
graphlab.cross_validation.shuffle(train_images)
# train_images = gl.cross_validation.shuffle(train_images)
test_images = graphlab.SFrame.read_csv('edges_only_test.csv')

features = train_images.column_names()
features.remove('class')

X_test, Y_test = get_numpy_data(test_images, features, 'class')



X,Y = get_numpy_data(train_images, features, 'class')
# Neural network experiment
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clf.fit(X,Y)
print get_residual_sum_of_squares(clf.predict(X_test), Y_test)

# SVM experiment
clf2 = svm.SVC()
clf2.fit(X, Y)
print get_residual_sum_of_squares(clf2.predict(X_test), Y_test)


# model = graphlab.classifier.create(train_images,target='class', features=features, validation_set=None)
# classification = model.classify(test_images)
# results = model.evaluate(test_images)
#
# print classification
# print results