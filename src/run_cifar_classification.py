import neural_net, data_utils
import numpy as np

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")
    nn = neural_net.TwoLayerNet(3072, 6144, 10, 0.01)

    # Find val set
    # Find good reg

    learning_rate = 0.1
    learning_rate_decay = 0.95
    reg = 0.008
    num_iters = 20000
    batch_size = 10000
    verbose = True

    print "Training ..."
    nn.train(X_train, y_train, X_test, y_test,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    print "Accuracy "
    print nn.accuracy(X_test,y_test)
