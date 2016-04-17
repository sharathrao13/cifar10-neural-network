import numpy as np
from random import randint
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        # self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        # self.params['b1'] = np.zeros(hidden_size)
        # self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        # self.params['b2'] = np.zeros(output_size)

        np.random.seed(0)
        self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.params['b2'] = np.zeros((1, output_size))

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        Z_layer1 = np.dot(X,W1)+b1
        A_layer1 = self.leaky_relu(Z_layer1)
        Z_layer2 = np.dot(A_layer1,W2)+b2
        scores = self.softmax(Z_layer2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################

        #diff = (scores.transpose() - y).transpose()
        #delta_output = self.replace_zero_with_small_value(np.square(diff))


        corect_logprobs = -np.log(scores[range(N), y])
        data_loss = np.sum(corect_logprobs)

        #data_loss = -np.sum(np.log(delta_output))
        #data_loss = data_loss/float(N)


        L2_regularization=reg*0.5*(np.sum(np.square(W1))+np.sum(np.square(W2)))

        print "Data Loss %s Regularization loss %s" %(data_loss,L2_regularization)

        loss = data_loss+L2_regularization
        loss = loss/float(N)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        #delta_output is examples x output size
        #Activation at layer 1 is examples x hidden_size
        #w2 is hidden_size x classes, so we need the transpose the Activation

        delta_output = scores
        delta_output[range(N), y] -= 1
        derivative_W2 = np.dot(np.transpose(A_layer1),delta_output)+reg*W2
        derivative_b2 = np.sum(delta_output,axis=0,keepdims=True    )

        dRelu = self.derivative_leaky_relu(A_layer1)
        delta_hidden = delta_output.dot(np.transpose(W2))*dRelu
        derivative_W1 = np.dot(np.transpose(X),delta_hidden)+reg*W1
        derivative_b1 = np.sum(delta_hidden)



        grads['W1'] =derivative_W1
        grads['W2'] =derivative_W2
        grads['b1'] =derivative_b1
        grads['b2'] =derivative_b2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            end = randint(batch_size,num_train-1)
            start = end -batch_size

            X_batch = X[start:end, :]
            y_batch = y[start:end]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            update_to_W1 = grads['W1']
            update_to_b1 = grads['b1']
            update_to_W2 = grads['W2']
            update_to_b2 = grads['b2']

            # print "Printing the min of update weights and biases "
            # print np.amin(update_to_W1)
            # print np.amin(update_to_b1)
            # print np.amin(update_to_W2)
            # print np.amin(update_to_b2)
            #
            # print "Printing the max of update weights and biases "
            # print np.amax(update_to_W1)
            # print np.amax(update_to_b1)
            # print np.amax(update_to_W2)
            # print np.amax(update_to_b2)


            W1 = self.params['W1']
            b1 = self.params['b1']
            W2 = self.params['W2']
            b2 = self.params['b2']

            W1+= -learning_rate*(update_to_W1)
            W2+= -learning_rate*(update_to_W2)
            b1+= -learning_rate*(update_to_b1)
            b2+= -learning_rate*(update_to_b2)

            # print "Printing the min of weights and biases "
            # print np.amin(W1)
            # print np.amin(b1)
            # print np.amin(W2)
            # print np.amin(b2)
            #
            # print "Printing the max of weights and biases "
            # print np.amax(W1)
            # print np.amax(b1)
            # print np.amax(W2)
            # print np.amax(b2)


            self.params['W1'] = W1
            self.params['b1'] = b1
            self.params['W2'] = W2
            self.params['b2'] = b2

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                print "Validation Accuracy: %s" %val_acc
                train_acc_history.append(train_acc)
                print "Training Accuracy %s " %train_acc
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################]
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        Z_layer1 = np.dot(X,W1)+b1
        A_layer1 = self.leaky_relu(Z_layer1)
        Z_layer2 = np.dot(A_layer1,W2)+b2

        scores = self.softmax(Z_layer2)
        y_pred = np.argmax(scores, axis=1)
        #print y_pred

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

    def accuracy(self, X, y):
        """
        Use the trained model to predict labels for X, and compute the accuracy.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
        - y: A numpy array of shape (N,) giving the correct labels.

        Returns:
        - acc: Accuracy

        """
        acc = (self.predict(X) == y).mean()

        return acc

    def relu(self, xw):
        for i in range(0, np.shape(xw)[0]):
            for j in range(0, np.shape(xw)[1]):
                xw[i][j] = max(0.0, xw[i][j])
                # print xw[i][j]
        return xw

    def leaky_relu(self, xw):
        for i in range(0, np.shape(xw)[0]):
            for j in range(0, np.shape(xw)[1]):
                xw[i][j] = max(0.0001*xw[i][j], xw[i][j])
                # print xw[i][j]
        return xw

    def softmax(self,X):
        print (np.amax(X))
        exponent = np.exp(X)
        sum_of_exponent = self.replace_zero_with_small_value(np.sum(exponent,axis=1, keepdims=True))
        #Collate everything to one axis so that every example has a softmax value
        softmax = exponent/sum_of_exponent
        return softmax

    def derivative_relu(self, X):
        for i in range(0, np.shape(X)[0]):
            for j in range(0, np.shape(X)[1]):
                if X[i][j]>0:
                    X[i][j] = 1
                else:
                    X[i][j] = 0
        return X

    def derivative_leaky_relu(self,X):
        for i in range(0, np.shape(X)[0]):
            for j in range(0, np.shape(X)[1]):
                if X[i][j]>0:
                    X[i][j] = 1
                else:
                    X[i][j] = 0.00000001
        return X

    def replace_zero_with_small_value(self, X):
        for i in range(0, np.shape(X)[0]):
            for j in range(0, np.shape(X)[1]):
                if(X[i][j]==0.0):
                    X[i][j]=0.00000001

        return X
