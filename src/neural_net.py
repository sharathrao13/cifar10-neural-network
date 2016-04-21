import numpy as np


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

    def __init__(self, input_size, hidden_size, output_size, std=1e-4, momentum=0.9):
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
        self.momentum = momentum

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.old_weights = {}
        self.old_weights['W1'] = self.params['W1']
        self.old_weights['b1'] = self.params['b1']
        self.old_weights['W2'] = self.params['W2']
        self.old_weights['b2'] = self.params['b2']

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
        #A_layer1, scores = self.feed_forward(X,W1,b1,W2,b2)

        Z1 = X.dot(W1) + b1
        A1 = self.Relu(Z1)
        Z2 = A1.dot(W2) + b2

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

        #Use this log(p/q) = logp-logq
        softmax_numerator = np.exp(Z2)
        softmax_denominator = np.sum(softmax_numerator, axis=1)
        data_loss = np.sum(-Z2[range(N), y] + np.log(softmax_denominator)) / N
        regularization_loss = self.calculate_L2_regularization(W1, W2, reg)

        loss = data_loss + regularization_loss

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

        d_A2 = 1.0
        d_Z2 = (softmax_numerator.T / softmax_denominator).T
        ground_truth = self.get_ground_truth(N, d_Z2, y)
        d_Z2 = ((d_Z2 - ground_truth)/float(N))*d_A2

        d_A1 = d_Z2.dot(W2.T)
        d_Z1 = d_A1 * self.derivative_relu(Z1)

        dW1 = X.T.dot(d_Z1)
        dW2 = A1.T.dot(d_Z2)
        db1 = np.sum(d_Z1, axis=0)
        db2 = np.sum(d_Z2, axis=0)

        dW1 += reg * W1
        dW2 += reg * W2

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

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
        best_val_acc = 0.0
        best_model = {}
        #To hold the best validation model, so that it can be used for
        #predictions

        for it in xrange(num_iters):

            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch,y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            W1 = self.params['W1']
            b1 = self.params['b1']
            W2 = self.params['W2']
            b2 = self.params['b2']

            #Use this for SGD
            # W1+= -learning_rate*(grads['W1'])
            # W2+= -learning_rate*(grads['W2'])
            # b1+= -learning_rate*(grads['b1'])
            # b2+= -learning_rate*(grads['b2'])

            #Use this for momentum
            self.old_weights['W1'] = self.momentum * self.old_weights['W1'] - learning_rate * grads['W1']
            self.old_weights['W2'] = self.momentum * self.old_weights['W2'] - learning_rate * grads['W2']
            self.old_weights['b1'] = self.momentum * self.old_weights['b1'] - learning_rate * grads['b1']
            self.old_weights['b2'] = self.momentum * self.old_weights['b2'] - learning_rate * grads['b2']

            W1+= self.old_weights['W1']
            W2+= self.old_weights['W2']
            b1+= self.old_weights['b1']
            b2+= self.old_weights['b2']

            self.params['W1'] = W1
            self.params['b1'] = b1
            self.params['W2'] = W2
            self.params['b2'] = b2

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Every epoch, check train and val accuracy and decay learning rate.
            if verbose and (it ==0 or (it) % (iterations_per_epoch*5) == 0):

                train_acc = self.find_mean(X_batch, y_batch)
                val_acc = self.find_mean(X_val, y_val)

                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                print "[Iteration %d / %d][Validation Accuracy: %s ][Training Accuracy %s ][Loss %s]" % (
                it, num_iters, val_acc, train_acc, loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model['W1'] = W1
                    best_model['b1'] = b1
                    best_model['W2'] = W2
                    best_model['b2'] = b2

            if (it+1) % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay

        #Set the global weights with best validation model weights
        self.params['W1'] = best_model['W1']
        self.params['b1'] = best_model['b1']
        self.params['W2'] = best_model['W2']
        self.params['b2'] = best_model['b2']

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

        scores = self.get_score(X)
        return np.argmax(scores, axis=1)

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

    def get_score(self, X):

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']


        Z1 = X.dot(W1) + b1
        A1 = self.Relu(Z1)
        Z2 = A1.dot(W2) + b2

        return Z2

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

        scores = self.get_score(X)

        score1 = np.argmax(scores,axis=1)
        for i in range(0,scores.shape[0]):
            scores[i][score1[i]] = -1;

        score2 = np.argmax(scores,axis=1)
        for i in range(0,scores.shape[0]):
            scores[i][score2[i]] = -1;

        score3 = np.argmax(scores,axis=1)

        top1 = (score1 == y).mean()
        top2 =0
        top3 =0

        for i in range (0,y.shape[0]):
            if (y[i]==score1[i]) or (y[i] ==score2[i]) or (y[i] ==score3[i]):
                top3+=1

            if (y[i]==score1[i]) or (y[i] ==score2[i]):
                top2+=1

        top2 = float(top2)/float(y.shape[0])
        top3 = float(top3)/float(y.shape[0])
        return (top1, top2, top3)

    def derivative_relu(self, Z1):
        return (Z1 >= 0)

    def Relu(self, Z1):
        return Z1 * (Z1 > 0)

    def derivative_tanh(self, tan_h):
        # Please note the value provided is tanh(X) and not X
        return (1 - np.power(tan_h, 2))

    def tanh(self, Z1):
        return np.tanh(Z1)

    def derivative_leaky_relu(self, Z1):
        for i in range (0, Z1.shape[0]):
            for j in range (0,Z1.shape[1]):
                if(Z1[i][j] <0):
                    Z1[i][j]= 0.01
        return Z1

    def leaky_relu(self, Z1):
        for i in range (0, Z1.shape[0]):
            for j in range (0,Z1.shape[1]):
                if(Z1[i][j] <0):
                    Z1[i][j]= 0.01*Z1[i][j]
        return Z1


    def calculate_L2_regularization(self, W1, W2, reg):
        return 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    def find_mean(self, X, y):
        return (self.predict(X) == y).mean()

    def get_ground_truth(self, N, d_Z2, y):
        ground_truth = np.zeros(d_Z2.shape)
        ground_truth[range(N), y] = 1
        return ground_truth