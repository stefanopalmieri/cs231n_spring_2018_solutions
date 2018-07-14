import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = np.dot(X[i], W)
    # first shift the values of scores so that the highest number is 0:
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))

    for j in range(num_classes):
      if (j == y[i]):
        dscore = probs[j] - 1
      else:
        dscore = probs[j]

      dW[:,j] += dscore * X[i]

    loss += -np.log(probs[y[i]])

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_examples = X.shape[0]

  scores = np.dot(X, W)
  # first shift the values of scores so that the highest number is 0:
  scores -= np.max(scores)
  probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  loss = -np.sum(np.log(probs[np.arange(num_examples),y]))
  loss /= num_examples
  loss += 0.5 * reg * np.sum(W * W)

  dscores = probs
  dscores[np.arange(num_examples),y] -= 1
  dscores /= num_examples

  dW = np.dot(X.T, dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

