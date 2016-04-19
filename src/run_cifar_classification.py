import neural_net, data_utils
import numpy as np

if __name__ == '__main__':

    input_size = 3072
    hidden_size = 500
    output_size =10

    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")
    nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001)

    #Configuration Parameters
    training_size =49000
    test_size = 1000
    validation_size = 1000

    learning_rate = 0.0001
    learning_rate_decay = 0.95
    reg = 1.0
    num_iters = 2000
    batch_size = 1000
    verbose = True

    # Subsample the data
    mask = range(training_size, training_size + validation_size)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(training_size)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(validation_size)
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


    nn.train(X_train, y_train, X_val, y_val,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    accuracy = nn.accuracy(X_test,y_test)
    print " Test Accuracy  ... %s " %(accuracy*100.0)
