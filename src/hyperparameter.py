import neural_net, data_utils
import numpy as np

if __name__ == '__main__':

    tuning_params={}

    #tuning_params['learning_rate']=[0.0001,0.00005, 0.00001]
    #tuning_params['reg']=[0.1,0.5,1.0,1.5]
    #tuning_params['hidden_size'] = [10,50,100,500,1000]
    #tuning_params['iterations'] =[5000,10000,20000,40000]
    #tuning_params['momentum'] =[0.5, 0.7, 0.85,0.9,0.95]
    #tuning_params['batch_size'] = [50,100,500,1000]

    input_size = 3072
    hidden_size = 500
    output_size =10
    momentum =0.99

    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10("C:\Users\SHARATH\Git\cs291k-mp1\dataset")

    #Configuration Parameters
    training_size =49000
    test_size = 1000
    validation_size = 1000

    learning_rate = 0.0001
    learning_rate_decay = 0.95
    reg = 0.01
    num_iters = 1000
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

    nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001, momentum)
    output = nn.train(X_train, y_train, X_val, y_val,
                      learning_rate, learning_rate_decay,
                      reg, num_iters,
                      batch_size, verbose)

    train_max = np.max(output['train_acc_history'])
    val_max = np.max(output['val_acc_history'])

    print "Training Accuracy %s" % train_max
    print "Validation Accuracy %s" % val_max

    accuracy = nn.accuracy(X_test, y_test)
    print " Test Accuracy  ... %s " % (accuracy * 100.0)

    # momentums = tuning_params['momentum']
    # print "Tuning Momentum "
    # print "***********************************************************************"
    # for momentum in momentums:
    #     print "Momentum %s"%momentum
    #     nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001, momentum)
    #     output = nn.train(X_train, y_train, X_val, y_val,
    #          learning_rate, learning_rate_decay,
    #          reg, num_iters,
    #          batch_size, verbose)
    #
    #     train_max = np.max(output['train_acc_history'])
    #     val_max = np.max(output['val_acc_history'])
    #
    #     print "Training Accuracy %s"%train_max
    #     print "Validation Accuracy %s"%val_max
    #
    #     accuracy = nn.accuracy(X_test,y_test)
    #     print " Test Accuracy  ... %s " %(accuracy*100.0)
    #
    # print "***********************************************************************"

    # iterations = tuning_params['iterations']
    # print "Tuning Iterations "
    # print "***********************************************************************"
    # for iter_current in iterations:
    #     print "Iterations %s"%iter
    #     nn = neural_net.TwoLayerNet(input_size, hidden_size, output_size, 0.00001, tuning_params['momentum'][0])
    #     output = nn.train(X_train, y_train, X_val, y_val,
    #          learning_rate, learning_rate_decay,
    #          tuning_params['reg'][0], iter_current,
    #          tuning_params['batch_size'][0], verbose)
    #
    #     train_max = np.max(output['train_acc_history'])
    #     val_max = np.max(output['val_acc_history'])
    #     print "Training Accuracy %s"%train_max
    #     print "Validation Accuracy %s"%val_max
    #
    #     accuracy = nn.accuracy(X_test,y_test)
    #     print " Test Accuracy  ... %s " %(accuracy*100.0)
    #
    # print "***********************************************************************"


    # #Learning Rate
    # learning_rates = tuning_params['learning_rate']
    # print "Tuning Learning Rates "
    # print "***********************************************************************"
    # for learning_rate in learning_rates:
    #     print "Learning Rate %s "%learning_rate
    #     nn = neural_net.TwoLayerNet(input_size, 500, output_size, 0.00001, tuning_params['momentum'][0])
    #     output = nn.train(X_train, y_train, X_val, y_val,
    #          learning_rate, learning_rate_decay,
    #          tuning_params['reg'][0], tuning_params['iterations'][0],
    #          tuning_params['batch_size'][0], verbose)
    #
    #     train_history = np.max(output['train_acc_history'])
    #     print "Training Accuracy %s"%train_history
    #
    #     accuracy = nn.accuracy(X_test,y_test)
    #     print " Test Accuracy  ... %s " %(accuracy*100.0)
    # print "***********************************************************************"