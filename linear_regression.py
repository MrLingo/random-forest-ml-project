import inline as inline
import matplotlib as matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.special import expit
import unittest

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def run_tests():
    unittest.main(argv=[''], verbosity=1, exit=False)


data = OrderedDict(
    amount_spent = [50,  10, 20, 5,  95,  70,  100,  200, 0],
    send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1]
)

df = pd.DataFrame.from_dict(data)

df.plot.scatter(x='amount_spent', y='send_discount');


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Sci-py function 'expit(z) do the same as the line above.'
# return expit(z)


class TestSigmoid(unittest.TestCase):

    def test_at_zero(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)

    def test_at_negative(self):
        self.assertAlmostEqual(sigmoid(-100), 0)

    def test_at_positive(self):
        self.assertAlmostEqual(sigmoid(100), 1)


run_tests()
'''
x = np.linspace(-10., 10., num=100)
sig = sigmoid(x)

plt.plot(x, sig, label="sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size' : 16})
plt.show()


def loss(h, y):
  return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

class TestLoss(unittest.TestCase):

  def test_zero_h_zero_y(self):
    self.assertLess(loss(h=0.000001, y=.000001), 0.0001)

  def test_one_h_zero_y(self):
    self.assertGreater(loss(h=0.9999, y=.000001), 9.0)

  def test_zero_h_one_y(self):
    self.assertGreater(loss(h=0.000001, y=0.9999), 9.0)

  def test_one_h_one_y(self):
    self.assertLess(loss(h=0.999999, y=0.999999), 0.0001)

run_tests()


X = df['amount_spent'].astype('float').values
y = df['send_discount'].astype('float').values

def predict(x, w):
  return sigmoid(x * w)

def print_result(y_hat, y):
  print(f'loss: {np.round(loss(y_hat, y), 5)} predicted: {y_hat} actual: {y}')

y_hat = predict(x=X[0], w=.5)
print_result(y_hat, y[0])

for w in np.arange(-1, 1, 0.1):
  y_hat = predict(x=X[0], w=w)
  print(loss(y_hat, y[0]))

  def fit(X, y, n_iter=1000, lr=0.01):
      W = np.zeros(X.shape[1])
      for i in range(n_iter):
          z = np.dot(X, W)
          h = sigmoid(z)
          gradient = np.dot(X.T, (h - y)) / y.size
          W -= lr * gradient
      return W

def predict(X, W):
    return sigmoid(np.dot(X, W))

# Insert code here.

class TestGradientDescent(unittest.TestCase):

    def test_correct_prediction(self):
        global X
        global y
        X = X.reshape(X.shape[0], 1)
        w = fit(X, y)
        y_hat = predict(X, w).round()
        self.assertTrue((y_hat == y).all())

run_tests()'''



  
