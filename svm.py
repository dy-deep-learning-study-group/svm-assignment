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
        return np.argmax(self.W.dot(X), axis=0)

        #######################################################################
        # TODO - Implement prediction.
        # Use argmax to return the label
        #######################################################################

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
        num_train = X.shape[1]

        scores = self.W.dot(X)

        correct_scores = scores[y, np.arange(0, scores.shape[1])]

        margins = np.maximum(0, (scores - correct_scores) + 1)

        loss = np.sum(margins) - num_train

        loss /= num_train
        loss += reg * np.sum(self.W**2)

        #######################################################################
        # TODO - Calculate the SVM gradient wrt W
        #
        # See lecture slides or README.md for relevant equations
        #
        # Hint - you shouldn't need to calcuate it from scratch, should be able
        # to reuse some values from loss calculation
        #######################################################################

        dW = np.zeros(self.W.shape)  # initialize the gradient as zero

        # Apply indicator function      # Apply indicator function      # Apply indicator function
        margins[margins < 0] = 0
        margins[margins > 0] = 1

        # y_i does not contribute to the loss
        margins[y, np.arange(0, len(y))] = 0
        # dW_i = number of classes that contrinuted to the loss * x_i
        margins[y, np.arange(0, len(y))] = -np.sum(margins, axis=0)

        dW = margins.dot(X.T)
        dW /= num_train

        dW += reg * 2 * self.W

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
            rand_idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[:, rand_idx]
            y_batch = y[rand_idx]

            loss, grad = self.compute_loss(X_batch, y_batch, reg)
            self.W -= learning_rate * grad

            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

        return loss_history
