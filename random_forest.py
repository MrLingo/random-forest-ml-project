from sklearn import tree
import pandas as pd
import random
import csv
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('house_prices_train.csv')
y = df_train['SalePrice']

csv_filename = 'house_prices_train.csv'
csv_test_filename = 'test.csv'
dataframe = pd.read_csv(csv_filename)
dataframe2 = pd.read_csv(csv_test_filename)

# Root mean square error
def rmse(h, y):
  print (sqrt(mean_squared_error(h, y)))

class RandomForest:
    def __init__(self, n_estimators):
        self.trees = []
        self.subsets = []
        self.tree_subset = []
        self.n_estimators = n_estimators
        self.feature = None
        self.feature_subset_size = 0
        self.dtree = None

    # Pick random values for each feature and subset size:
    def get_random_subset(self, attr):
        self.feature_subset_size = random.randrange(2, len(attr))
        for x in range(self.feature_subset_size):
            self.feature = random.choice(attr)
            self.tree_subset.append(self.feature)
        return self.tree_subset

    def fit(self, X, y):
        for i in range(self.n_estimators):
            subset = self.get_random_subset(list(X))
            self.tree_subset = []
            self.subsets.append(subset)
            self.dtree = tree.DecisionTreeClassifier()
            self.dtree.fit(X[subset], y)
            self.trees.append(self.dtree)

    def predict_rf(self, x):
        # Avoiding 'index out range' error
        ans = []
        for i in range(len(x)):
            ans.append(0)

        treesNum = len(self.trees)
        # Get array ( prediction ) from each tree and assign each value from it in the corresponding ans element
        for i in range(treesNum):
            tree_prediction = self.trees[i].predict(x[self.subsets[i]])
            for j in range(len(tree_prediction)):
                ans[j] += tree_prediction[j]

        # Get average
        for k in range(len(ans)):
            ans[k] /= treesNum

        return ans


# Training set. Every column that contains only quantitative data.
X = dataframe[['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 'BedroomAbvGr',
               'FullBath', 'YrSold', 'MiscVal', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
               'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'GarageCars', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]

# Run the forest with 10 trees
random_forest = RandomForest(10)
random_forest.fit(X, y)
preds = random_forest.predict_rf(X)

metrics.r2_score(y, preds)
rmse(preds, y)

# Test set.
X_test = dataframe2[['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 'BedroomAbvGr',
               'FullBath', 'YrSold', 'MiscVal', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
               'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'GarageCars', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]


X_test = X_test.fillna(0)
pred_test = random_forest.predict_rf(X_test)
#print(pred_test)

# Sending to kaggle
submission = pd.DataFrame({'Id': dataframe2.Id, 'SalePrice': pred_test})
print(submission)
