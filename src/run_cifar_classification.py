import neural_net, data_utils
import numpy as np

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")
    nn = neural_net.TwoLayerNet(3072, 4000, 10, 0.0001)

    # Find val set
    # Find good reg

    learning_rate = 0.0000001
    learning_rate_decay = 0.50
    reg = 0.0000001
    num_iters = 2
    batch_size = 10000
    verbose = True

    print "Normalizing Image "
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    mean_image_value_train = np.mean(X_train, axis=0)
    X_train -= mean_image_value_train
    mean_image_value_test = np.mean(X_test, axis=0)
    X_test -=mean_image_value_test

    X_val = X_train[2000:3000,:]
    y_val = y_train[2000:3000]

    print "Training ..."
    nn.train(X_train, y_train, X_val, y_val,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    print "Calculating Accuracy "
    print nn.accuracy(X_test,y_test)
