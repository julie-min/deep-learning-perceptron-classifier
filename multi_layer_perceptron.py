import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from data_preparation import prepare_data

def visualize(net, X, y, labels, colors, xlabel, ylabel):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = net.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    for i, color in zip(range(len(labels)), colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("MLP Classification")
    plt.show()

if __name__ == "__main__":
    nSamples = 1000
    X, y, labels = prepare_data(nSamples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    print(f"Training accuracy: {mlp.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {mlp.score(X_test, y_test):.4f}")

    visualize(mlp, X, y, labels, ['blue', 'red'], 'X', 'Y')