import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self):
        self.tree = None

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        base_gini = self.gini_impurity(y)

        for feature in X.columns:
            for threshold in np.unique(X[feature]):
                left_idx = X[feature] <= threshold
                right_idx = ~left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                left_gini = self.gini_impurity(y[left_idx])
                right_gini = self.gini_impurity(y[right_idx])
                weighted_gini = (left_idx.sum() * left_gini + right_idx.sum() * right_gini) / len(y)
                gain = base_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build(self, X, y):
        if len(np.unique(y)) == 1:
            return y.iloc[0]

        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            return y.mode()[0]

        left = X[feature] <= threshold
        right = ~left

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build(X[left], y[left]),
            'right': self.build(X[right], y[right])
        }

    def fit(self, X, y):
        self.tree = self.build(X, y)

    def predict_sample(self, row, node):
        if not isinstance(node, dict):
            return node

        if row[node['feature']] <= node['threshold']:
            return self.predict_sample(row, node['left'])
        else:
            return self.predict_sample(row, node['right'])

    def predict(self, X):
        return X.apply(lambda row: self.predict_sample(row, self.tree), axis=1)

def compute_accuracy(y_true, y_pred):
    return 100 * np.mean(y_true == y_pred)

data = pd.read_csv('iris.csv')

data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_point = int(0.8 * len(data))
train, test = data[:split_point], data[split_point:]

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

tree = DecisionTree()
tree.fit(X_train, y_train)
predictions = tree.predict(X_test).reset_index(drop=True)
actual = y_test.reset_index(drop=True)

results = pd.DataFrame({
    'Model Prediction': predictions,
    'Actual Class': actual
})

print(results)
print("\nAccuracy: {:.2f}%".format(compute_accuracy(actual, predictions)))
