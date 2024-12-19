import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load Spambase dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None)

# Prepare features and labels
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values    # Labels (0 for non-spam, 1 for spam)

print("Number of samples:", len(df))
print('Features:', X.shape)
print('Labels:', y.shape)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=0),
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Gradient Boosting': GradientBoostingClassifier(random_state=0)
}

# Function to train models and print accuracy
def train_and_evaluate(models, X_train, y_train, description=""):
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_train, y_train) * 100
        print(f'{name} Accuracy ({description}): {acc:.2f}%')

# Evaluate without PCA
train_and_evaluate(models, X_scaled, y, "no compression")

print('==================================');

# Principal Component Analysis (PCA)
# Compute PCA
mean = np.mean(X_scaled, axis=0)
X_bar = X_scaled - mean
cov = np.matmul(X_bar.transpose(), X_bar)
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Project data onto the first k principal components
k = 10
P = eigenvectors[:, :k]
X_pca = np.matmul(X_bar, P)

# Evaluate with PCA
train_and_evaluate(models, X_pca, y, "with PCA compression")
print('We compress:', X.shape[1] / k, 'times')

# PCA Visualization
labels = ['Non-Spam', 'Spam']
plt.figure(figsize=(8, 6))
for l in range(2):
    plt.scatter(X_pca[y == l, 0], X_pca[y == l, 1], label=labels[l])
plt.title('Principal Component Analysis')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.savefig('PCA.png')
plt.show()

# t-SNE Visualization
X_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X_pca)
plt.figure(figsize=(8, 6))
for l in range(2):
    plt.scatter(X_tsne[y == l, 0], X_tsne[y == l, 1], label=labels[l])
plt.title('t-SNE visualization')
plt.xlabel('tSNE 1')
plt.ylabel('tSNE 2')
plt.legend()
plt.savefig('tSNE.png')
plt.show()

print('Done')
