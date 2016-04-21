#!/usr/bin/python

import sys
import neural_net, data_utils
import numpy as np
import time

if __name__ == '__main__':

    start_time = time.time()
    #C:\Users\SHARATH\Git\cs291k-mp1\dataset
    file_location = sys.argv[1]+"/cifar-10-batches-py"
    print file_location

    input_size = 3072
    hidden_size = 500
    output_size =10
    momentum =0.95

    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(file_location)
    nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001, momentum)

    #Configuration Parameters
    training_size =49000
    test_size = 10000
    validation_size = 1000

    learning_rate = 0.0001
    learning_rate_decay = 0.95
    reg = 0.01
    num_iters = 200
    batch_size = 500
    verbose = True

    mask = range(training_size, training_size + validation_size)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(training_size)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(test_size)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.reshape(training_size, -1)
    X_val = X_val.reshape(validation_size, -1)
    X_test = X_test.reshape(test_size, -1)


    output = nn.train(X_train, y_train, X_val, y_val,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    print "********************************************************************"
    best_train_acc = np.max(output['train_acc_history'])
    print "Training Accuracy %s" % (best_train_acc*100.0)
    best_val_acc = np.max(output['val_acc_history'])
    print "Validation Accuracy %s" % (best_val_acc*100.0)

    accuracy = nn.accuracy(X_test,y_test)
    print "********************************************************************"
    print " Test Accuracy Top 1 ... %s " %(accuracy[0]*100.0)
    print " Test Accuracy Top 2 ... %s " %(accuracy[1]*100.0)
    print " Test Accuracy Top 3 ... %s " %(accuracy[2]*100.0)

    print "********************************************************************"
    print"Took %s seconds" % (time.time() - start_time)
