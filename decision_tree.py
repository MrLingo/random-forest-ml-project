import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest
import math
from sklearn import metrics
from sklearn.tree import export_graphviz
import IPython, graphviz, re

%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    Source from: https://github.com/fastai/fastai/blob/e6b56de53f80d2b2d39037c82d3a23ce72507cd7/old/fastai/structured.py#L22
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))
       
       
       
df_train = pd.read_csv('house_prices_train.csv')

X = df_train[['OverallQual', 'GrLivArea', 'GarageCars']]
y = df_train['SalePrice']

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(h, y):
  return sqrt(mean_squared_error(h, y))
  
  
from sklearn.ensemble import RandomForestRegressor

# max_depth parameter - maximum level of depthness if needed.
reg = RandomForestRegressor(n_estimators=1,  bootstrap=False, random_state=RANDOM_SEED)

reg.fit(X, y)  

draw_tree(reg.estimators_[0], X, precision=2)

preds = reg.predict(X)

rmse(preds, y)

class Node:
  
  def __init__(self, x, y, idxs, min_leaf=5):
    self.x = x
    self.y = y
    self.idxs = idxs
    self.min_leaf = min_leaf
    
    self.row_count = len(idxs)
    self.col_count = x.shape[1]
    self.val = np.mean(y[idxs])
    self.score = float('inf')
    self.find_varsplit()
    
  def find_varsplit(self):
    for c in range(self.col_count): self.find_better_split(c)
    if self.is_leaf: return
    x = self.split_col
    lhs = np.nonzero(x <= self.split)[0]
    rhs = np.nonzero(x > self.split)[0]
    self.lhs = Node(self.x, self.y, self.idxs[lhs])
    self.rhs = Node(self.x, self.y, self.idxs[rhs])
    
  def find_better_split(self, var_idx):
    x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
    
    for r in range(self.row_count):
      lhs = x <= x[r]
      rhs = x > x[r]
      if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf : continue
      lhs_std = y[lhs].std()
      rhs_std = y[rhs].std()
      curr_score = lhs_std * lhs.sum() + rhs_std * rhs.sum() # Weighted average
      if curr_score < self.score:
        self.var_idx = var_idx
        self.score = curr_score
        self.split = x[r]

  @property
  def is_leaf(self):
    return self.score == float('inf')
  
  @property
  def split_col(self): return self.x.values[self.idxs, self.var_idx]
  
  def predict(self, x):
    return np.array([self.predict_row(xi) for xi in x])
  
  def predict_row(self, xi):
    if self.is_leaf: return self.val
    n = self.lhs if xi[self.var_idx] <= self.split else self.rhs
    return n.predict_row(xi)
    
 class DecisionTreeRegressor:
  def fit(self, X, y, min_leaf=5):
    # Sub set of the whole dataset for each branch np.arange(len(y))
    self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
    return self
  
  def predict(self, X):
    return self.dtree.predict(X.values)
    
 regressor = DecisionTreeRegressor().fit(X, y)
 preds = regressor.predict(X)   
 
 metrics.r2_score(y, preds)
 
 rmse(preds, y)
