import neural_net, data_utils
import numpy as np
import matplotlib.pyplot as plt
import time
if __name__ == '__main__':

    start_time = time.time()

    input_size = 3072
    hidden_size = 500
    output_size =10
    momentum =0.95

    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")
    nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001, momentum)

    #Configuration Parameters
    training_size =49000
    test_size = 10000
    validation_size = 1000

    learning_rate = 0.0001
    learning_rate_decay = 0.95
    reg = 0.01
    num_iters = 20000
    batch_size = 500
    verbose = True

    # Subsample the data
    mask = range(training_size, training_size + validation_size)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(training_size)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(test_size)
    X_test = X_test[mask]
    y_test = y_test[mask]


    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(training_size, -1)
    X_val = X_val.reshape(validation_size, -1)
    X_test = X_test.reshape(test_size, -1)


    output = nn.train(X_train, y_train, X_val, y_val,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    print "********************************************************************"
    best_train_acc = np.max(output['train_acc_history'])
    print "Training Accuracy %s" % best_train_acc
    best_val_acc = np.max(output['val_acc_history'])
    print "Validation Accuracy %s" % best_val_acc

    accuracy = nn.accuracy(X_test,y_test)
    print "********************************************************************"
    print " Test Accuracy Top 1 ... %s " %(accuracy[0])
    print " Test Accuracy Top 2 ... %s " %(accuracy[1])
    print " Test Accuracy Top 3 ... %s " %(accuracy[2])

    print "********************************************************************"
    print"Took %s seconds" % (time.time() - start_time)
