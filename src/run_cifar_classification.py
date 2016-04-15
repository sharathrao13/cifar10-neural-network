import neural_net, data_utils
import numpy as np

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")
    nn = neural_net.TwoLayerNet(3072, 6144, 10, 0.01)

    # Find val set
    # Find good reg

    learning_rate = 0.001
    learning_rate_decay = 0.95
    reg = 0.00001
    num_iters = 1
    batch_size = 100
    verbose = True

    print "Training ..."
    nn.train(X_train, y_train, X_test, y_test,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    print nn.accuracy(X_test,y_test)