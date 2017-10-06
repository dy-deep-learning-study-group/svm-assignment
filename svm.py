import numpy as np


class SupportVectorMachine:
    def __init__(self):
        self.W = None

    def predict(self, X):
        '''
        Inputs:
            X - Matrix of data points of dimensions (D, N)
                D - dimensions of data
                N - number of data points
        Returns:
            The predicted class
        '''
        #######################################################################
        # TODO - Implement prediction.
        # Use argmax to return the label
        #######################################################################
        pass

    def compute_loss(self, X, y, reg):
        '''
        Inputs:
            X             - Matrix of training data points of dimensions (D, N)
            y             - Numeric labels for training data of dimensions (N,)
            reg           - Regularisation strength
        Returns:
            loss          - The SVM loss
            dW            - Gradient wrt to W
        '''
        #######################################################################
        # TODO - Calculate the SVM 'Hinge loss'
        # See lecture slides or README.md for the relevant equations.
        #
        # Hint - you should be able to program this without need for any loops.
        # You should be able to do it with numpy vectorized optimizations
        #######################################################################
        loss = 0.0

        #######################################################################
        # TODO - Calculate the SVM gradient wrt W

        # See lecture slides or README.md for relevant equations
        #
        # Hint - you shouldn't need to calcuate it from scratch, should be able
        # to reuse some values from loss calculation
        #######################################################################

        dW = np.zeros(self.W.shape)  # initialize the gradient as zero

        return loss, dW

    def train(self, X, y, learning_rate, reg,
              num_iters, batch_size, verbose=False):
        '''
        Inputs:
            X             - Matrix of training data points of dimensions (D, N)
            y             - Numeric labels for training data of dimensions (N,)
            learning_rate - The learning rate
            reg           - Regularisation strength
            num_iters     - Number of iterations
            batch_size    - Minibatch size for stochastic gradient descent
        Returns:
            loss_history  - list of calculate loss values
        '''

        dim, num_train = X.shape
        num_classes = np.max(y) + 1

        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(0, num_iters):
            ###################################################################
            # TODO - Implement Stochastic Gradient descent
            # Randomly sample X and y, to calculate loss.
            # See numpy.random.choice
            ###################################################################
            loss = 0.0

            # Your code here

            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

        return loss_history
