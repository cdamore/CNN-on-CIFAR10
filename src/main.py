import tensorflow as tf
import numpy as np
import timeit
from net import train, test
from iterator import DatasetIterator
from collections import OrderedDict
from pprint import pformat


if __name__ == '__main__':
    # load CIFAR10 dataset
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    batch_size = 64
    # change TRAIN to true if you want to create a new model and save it to the ckpt folder
    TRAIN = False
    if TRAIN:
        train(x_train, y_train, batch_size)
    cifar10_test = DatasetIterator(x_test, y_test, batch_size)
    cifar10_test_images, cifar10_test_labels = x_test, y_test

    # start timer
    start = timeit.default_timer()
    np.random.seed(0)
    # get results from test set
    predicted_cifar10_test_labels = test(cifar10_test_images)
    np.random.seed()
    # end timer
    stop = timeit.default_timer()
    run_time = stop - start
    # calculate accuracy
    correct_predict = (cifar10_test_labels.flatten() == predicted_cifar10_test_labels.flatten()).astype(np.int32).sum()
    incorrect_predict = len(cifar10_test_labels) - correct_predict
    accuracy = float(correct_predict) / len(cifar10_test_labels)
    print('Acc: {}. Testing took {}s.'.format(accuracy, stop - start))

    result = OrderedDict(correct_predict=correct_predict,
                     accuracy=accuracy,
                     run_time=run_time)
    # save results
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result, indent=4))
    print(pformat(result, indent=4))
