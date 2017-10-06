import pickle
import sys
import gzip
import matplotlib.pyplot as plt
import numpy as np
from colors import printc, BLUE, GREEN

from svm import SupportVectorMachine

def load_dataset():
    print("Loading mnist dataset...")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        if sys.version_info > (3,0):
            return pickle.load(f, encoding='latin-1')
        else:
            return pickle.load(f)

def save_trained_svm(svm, filename):
    with open(filename, 'wb') as f:
        pickle.dump(svm, f)
        print("SVM model serialized to file {}".format(filename))

def load_trained_svm(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_X_and_y(dataset):
    X = dataset[0]
    # Add additional row of ones for the bais
    X = np.hstack([X, np.ones((X.shape[0], 1))]).T
    y = dataset[1].T
    return X, y

def visualize_svm_weights(svm):
    plt.clf()
    w = svm.W[:, :-1]  # strip out the bias",
    w_min, w_max = np.min(w), np.max(w)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # Rescale the weights to be between 0 and 255\n",
        wimg = 255.0 * (w[i, :].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(np.reshape(wimg, [28, 28]), cmap='gray')
        plt.axis('off')
        plt.title(i)
    plt.draw()
    plt.waitforbuttonpress()


def tune_hyperparameters(X_train, y_train, X_valid, y_valid):
    pass

def run_live_classification(X_test, y_test, svm):
    print("Running live classification.\n"
          "Press any key to advance to next random image, close figure to quit")

    dim, num_test = X_test.shape

    while plt.get_fignums():
        idx = np.random.choice(num_test)
        img = X_test[:, idx]
        label = y_test[idx]

        prediction = svm.predict(img)

        plt.clf()
        plt.title("Actual value: {}, Predicted: {}".format(label, prediction))
        plt.imshow(np.reshape(img[:-1], [28, 28]), cmap='gray')
        plt.draw()
        plt.waitforbuttonpress(5)


if __name__ == '__main__':

    train_set, valid_set, test_set = load_dataset()
    X_train, y_train = get_X_and_y(train_set)

    ###########################################################################
    # TODO - Tune these hyperparameters
    # How you do this is up to you, but you may want to fill out the  function
    # tune_hyperparmeters to automatically try lots of different values
    ###########################################################################
    hyperparameters_tuned = False
    rate = 0           # learning rate
    batch_size = 10    # size of your minibatch. Don't be afraid to try larger sizes
    reg = 0            # Regularization strength. If poorly set, may get overflow
    num_iters = 1      # Number of iterations for stochastic gradient descent

    if not hyperparameters_tuned:
        print("Tuning hyperparameters!")
        X_valid, y_valid = get_X_and_y(valid_set)
        tune_hyperparameters(X_train, y_train, X_valid, y_valid)
    else:

        #######################################################################
        # NOTE - If you already have a serialized pre-trained SVM, you can load
        # it with load_trained_svm
        #######################################################################

        svm = SupportVectorMachine()
        loss_history = svm.train(X_train,
                                 y_train,
                                 learning_rate=rate,
                                 batch_size=batch_size,
                                 reg=reg,
                                 verbose=True,
                                 num_iters=num_iters)

        #######################################################################
        # NOTE - Training takes a long time, you may wish to save and
        # serialize your trained SVM for later use with save_trained_svm
        #######################################################################

        plt.plot(range(0, len(loss_history)), loss_history)
        plt.draw()
        plt.waitforbuttonpress()

        # Visualize the learned weights for each class
        visualize_svm_weights(svm)

        # Test time!
        predicted = svm.predict(X_train)
        printc(GREEN, "Training accuracy: {}"
                      .format(np.mean(predicted == y_train)))

        X_test, y_test = get_X_and_y(test_set)
        test_pred = svm.predict(X_test)

        printc(BLUE, "Test accuracy: {}"
                     .format(np.mean(test_pred == y_test)))

        run_live_classification(X_test, y_test, svm)
