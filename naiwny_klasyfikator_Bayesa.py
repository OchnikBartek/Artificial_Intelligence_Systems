import pandas as pd
import numpy as np

dataset = pd.read_csv('iris.csv')

shuffled = dataset.sample(frac=1.0, random_state=123).reset_index(drop=True)
cutoff = int(0.8 * len(shuffled))
train_data, test_data = shuffled[:cutoff], shuffled[cutoff:]

X_tr, y_tr = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_te, y_te = test_data.iloc[:, :-1], test_data.iloc[:, -1]

def summarize_by_class(features, labels):
    summary = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        subset = features[labels == lbl]
        summary[lbl] = (subset.mean(), subset.var())
    return summary

def normal_distribution(val, mu, sigma2):
    epsilon = 1e-9
    denominator = np.sqrt(2 * np.pi * (sigma2 + epsilon))
    exponent = np.exp(-((val - mu) ** 2) / (2 * (sigma2 + epsilon)))
    return exponent / denominator

def classify(instance, model_stats, class_priors):
    best_score = -np.inf
    selected = None
    for cls in model_stats:
        mu, sigma2 = model_stats[cls]
        log_likelihood = np.log(class_priors[cls])
        log_likelihood += np.sum(np.log(normal_distribution(instance, mu, sigma2)))
        if log_likelihood > best_score:
            best_score = log_likelihood
            selected = cls
    return selected

def run_prediction(input_set, train_X, train_y):
    stats = summarize_by_class(train_X, train_y)
    priors = train_y.value_counts(normalize=True).to_dict()
    return input_set.apply(lambda row: classify(row, stats, priors), axis=1)

def compute_accuracy(true_vals, predicted_vals):
    return 100 * np.mean(true_vals == predicted_vals)

predicted = run_prediction(X_te, X_tr, y_tr).reset_index(drop=True)
actual = y_te.reset_index(drop=True)

results_df = pd.DataFrame({
    'Model Prediction': predicted,
    'Actual Class': actual})

acc = compute_accuracy(actual, predicted)
print(results_df)
print("\nAccuracy: {:.2f}%".format(acc))
