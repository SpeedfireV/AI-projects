import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


data = pd.read_csv("winequality-red.csv", sep=";")
print(data.head())


data["quality_bin"] = (data["quality"] >= 6).astype(int)
data = data.drop(columns=["quality"])


X = data.drop(columns=["quality_bin"]).values
y = data["quality_bin"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def gini_impurity(y):
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def best_split(X, y):
    best_feature, best_threshold = None, None
    best_gain = 0
    current_impurity = gini_impurity(y)

    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            impurity = (len(left)/n_samples) * gini_impurity(left) + \
                       (len(right)/n_samples) * gini_impurity(right)
            gain = current_impurity - impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def build(self, X, y, depth=0):
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))

        feature, threshold, gain = best_split(X, y)

        if gain == 0:
            return np.argmax(np.bincount(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build(X[left_mask], y[left_mask], depth + 1),
            "right": self.build(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self.build(X, y)

    def predict_sample(self, node, x):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_sample(node["left"], x)
        else:
            return self.predict_sample(node["right"], x)

    def predict(self, X):
        return np.array([self.predict_sample(self.tree, x) for x in X])


tree = DecisionTree(max_depth=6, min_samples_split=10)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\n=== DECISION TREE (train/test) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))



def cross_val_score_custom_tree(X, y, max_depth=6, min_samples_split=10, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        tree.fit(X_train_cv, y_train_cv)
        y_pred_cv = tree.predict(X_val_cv)
        scores.append(accuracy_score(y_val_cv, y_pred_cv))

    return np.array(scores)

tree_scores = cross_val_score_custom_tree(
    X, y, max_depth=6, min_samples_split=10, n_splits=5
)

print("\n=== CROSS-VALIDATION: CUSTOM DECISION TREE ===")
print("Fold accuracies:", tree_scores)
print("Mean accuracy:", tree_scores.mean())
print("Std deviation:", tree_scores.std())



logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

y_pred_lr = logreg.predict(X_test)

print("\n=== LOGISTIC REGRESSION (train/test) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))



logreg_cv = LogisticRegression(max_iter=5000)
scores_lr = cross_val_score(logreg_cv, X, y, cv=5)

print("\n=== CROSS-VALIDATION: LOGISTIC REGRESSION ===")
print("Fold accuracies:", scores_lr)
print("Mean accuracy:", scores_lr.mean())
print("Std deviation:", scores_lr.std())

depths = range(1, 15)
accuracies = []

for d in depths:
    tree = DecisionTree(max_depth=d, min_samples_split=5)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(7,5))
plt.plot(depths, accuracies, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Impact of max depth of tree on accuracy")
plt.grid(True)
plt.show()
