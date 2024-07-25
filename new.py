import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from pyswarm import pso
import warnings

# Ignore warnings for simplicity
warnings.filterwarnings('ignore')

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class DBN:
    def __init__(self, rbm_layers=[256, 128], learning_rate=0.01, n_iter=10):
        self.rbm_layers = rbm_layers
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rbms = []
        self.logistic = LogisticRegression(max_iter=10000)

    def fit(self, X, y):
        input_data = X
        for n_components in self.rbm_layers:
            rbm = BernoulliRBM(n_components=n_components, learning_rate=self.learning_rate, n_iter=self.n_iter)
            rbm.fit(input_data)
            self.rbms.append(rbm)
            input_data = rbm.transform(input_data)
        
        self.logistic.fit(input_data, y)

    def transform(self, X):
        transformed_data = X
        for rbm in self.rbms:
            transformed_data = rbm.transform(transformed_data)
        return transformed_data

    def predict(self, X):
        transformed_data = self.transform(X)
        return self.logistic.predict(transformed_data)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def objective_function(params):
    learning_rate, n_components1, n_components2 = params
    dbn = DBN(rbm_layers=[int(n_components1), int(n_components2)], learning_rate=learning_rate, n_iter=10)
    dbn.fit(X_train, y_train)
    accuracy = dbn.score(X_test, y_test)
    return -accuracy

# Define bounds for PSO
lb = [0.001, 64, 32]
ub = [0.1, 512, 256]

# Run PSO
optimal_params, optimal_accuracy = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)
print(f"Optimal Parameters: {optimal_params}, Optimal Accuracy: {-optimal_accuracy}")

# Train the DBN with optimal parameters from PSO
optimal_learning_rate, optimal_n_components1, optimal_n_components2 = optimal_params
dbn = DBN(rbm_layers=[int(optimal_n_components1), int(optimal_n_components2)], learning_rate=optimal_learning_rate, n_iter=10)
dbn.fit(X_train, y_train)

# Predictions
y_train_pred = dbn.predict(X_train)
y_test_pred = dbn.predict(X_test)

# Calculate metrics
train_accuracy_dbn = accuracy_score(y_train, y_train_pred)
test_accuracy_dbn = accuracy_score(y_test, y_test_pred)
train_precision_dbn = precision_score(y_train, y_train_pred)
test_precision_dbn = precision_score(y_test, y_test_pred)

print(f"DBN Train Accuracy: {train_accuracy_dbn}")
print(f"DBN Test Accuracy: {test_accuracy_dbn}")
print(f"DBN Train Precision: {train_precision_dbn}")
print(f"DBN Test Precision: {test_precision_dbn}")

# Plotting the results
results = {
    'Metric': ['Accuracy', 'Precision'],
    'Train': [train_accuracy_dbn, train_precision_dbn],
    'Test': [test_accuracy_dbn, test_precision_dbn]
}

results_df = pd.DataFrame(results)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Train Metrics
ax[0].bar(results_df['Metric'], results_df['Train'], color=['blue', 'green'])
ax[0].set_title('Train Metrics Comparison')
ax[0].set_ylabel('Score')
ax[0].set_ylim([0, 1])
for i, v in enumerate(results_df['Train']):
    ax[0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# Test Metrics
ax[1].bar(results_df['Metric'], results_df['Test'], color=['blue', 'green'])
ax[1].set_title('Test Metrics Comparison')
ax[1].set_ylabel('Score')
ax[1].set_ylim([0, 1])
for i, v in enumerate(results_df['Test']):
    ax[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
