import numpy as np

def get_numpy_data(data_sframe, features, output):
    '''
    :param data_sframe: raw sframe object
    :param features: list of feature names (string)
    :param output: name of the target output (string_
    :return: tuple of numpy arrays
    '''
    data_sframe['constant'] = 1  # add 1's as constant feature
    features = ['constant'] + features  # list of features knows there's a constant one
    feature_matrix = data_sframe[features].to_numpy()
    output_array = data_sframe[output].to_numpy()
    return feature_matrix, output_array

def get_residual_sum_of_squares(predictions, outcome):
    # Then compute the residuals/errors
    residuals = predictions - outcome
    # Then square and add them up
    RSS = (residuals * residuals).sum()
    return RSS


def shuffle_lines(file):
    import csv
    import random

    with open(file) as f:
        r = csv.reader(f)
        header, l = next(r), list(r)

    a = [x[0] for x in l]
    random.shuffle(a)

    b = [x[1] for x in l]
    random.shuffle(b)

    with open(file, "wb") as f:
        csv.writer(f).writerows([header] + zip(a, b))