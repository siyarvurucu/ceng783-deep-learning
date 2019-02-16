import cPickle
import gzip
import scipy.io as sio # you may use scipy for loading your data
from math import sqrt, ceil
import numpy as np
from random import randrange
import os

def load_mnist(path = '../data/mnist.pkl.gz'):
    "Load the MNIST dataset and return training, validation and testing data separately"
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    X_train, y_train = training_data[0], training_data[1]
    X_val, y_val = validation_data[0], validation_data[1]
    X_test, y_test = test_data[0], test_data[1]

    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
def eval_numerical_gradient(f, x, verbose=False):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    x[ix] += h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] -= h # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad

def grad_check_sparse(f, x, analytic_grad, num_checks):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """
  h = 1e-5

  x.shape
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    x[ix] += h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] -= 2 * h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] += h # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
    
def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 3D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (D, D, H)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (D, D, H) = Xs.shape
  grid_size = int(ceil(sqrt(H)))
  grid_height = D * grid_size + padding * (grid_size - 1)
  grid_width = D * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width))
  next_idx = 0
  y0, y1 = 0, D
  for y in xrange(grid_size):
    x0, x1 = 0, D
    for x in xrange(grid_size):
      if next_idx < H:
        img = Xs[:, :, next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += D + padding
      x1 += D + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid



def load_BP_dataset(filename):
        """Load your 'PPG to blood pressure' dataset"""
        # TODO: Fill this function so that your version of the data is loaded from a file into vectors
        import hdf5storage  # got error with scipy.io 
        mat = hdf5storage.loadmat(filename)
        data=mat[u'Part_1']
        X = []
        Y = []
        for i in range(3000):
            X = np.append(X,data[0,i][0,:])  # load PPG data
            Y = np.append(Y,data[0,i][1,:])  # load ABP data
        pass
	# END OF YOUR CODE


        return X, Y

