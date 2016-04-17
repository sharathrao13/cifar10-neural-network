import neural_net, data_utils
import numpy as np

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("/cs/student/sharathrao/cs291k-mp1/dataset")
    nn = neural_net.TwoLayerNet(3072, 4000, 10, 0.0001)

    # Find val set
    # Find good reg
    file = open('output_run', 'w')
    learning_rate = 0.001
    learning_rate_decay = 0.95
    reg = 0.001
    num_iters = 2
    batch_size = 1000
    verbose = True

    file.write ('Normalizing Image \n')

    X_train = X_train.astype(float)/255.0
    X_test = X_test.astype(float)/255.0
    mean_image_value_train = np.mean(X_train, axis=0)
    X_train -= mean_image_value_train
    mean_image_value_test = np.mean(X_test, axis=0)
    X_test -=mean_image_value_test

    X_val = X_train[2000:3000,:]
    y_val = y_train[2000:3000]

    file.write ('Training ... \n')
    nn.train(X_train, y_train, X_val, y_val,
             learning_rate, learning_rate_decay,
             reg, num_iters,
             batch_size, verbose)

    accuracy = nn.accuracy(X_test,y_test)
    file.write ('Accuracy  ... %s'%accuracy)
