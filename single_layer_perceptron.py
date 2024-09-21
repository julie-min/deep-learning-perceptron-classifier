import numpy as np
import matplotlib.pyplot as plt
from data_preparation import prepare_data

def step(x):
    return int(x <= 0)

class Perceptron():
    def __init__(self, dim, activation):
        rnd = np.random.default_rng()
        self.dim = dim
        self.activation = activation
        self.w = rnd.normal(scale=np.sqrt(2.0 / dim), size=dim)
        self.b = rnd.normal(scale=np.sqrt(2.0 / dim))

    def printW(self):
        for i in range(self.dim):
            print('  w{} = {:6.3f}'.format(i + 1, self.w[i]), end='')
        print('  b = {:6.3f}'.format(self.b))

    def predict(self, x):
        return np.array([self.activation(np.dot(self.w, x[i]) + self.b)
                         for i in range(len(x))])

    def fit(self, X, y, N, epochs, eta=0.01):
        idx = list(range(N))
        np.random.shuffle(idx)
        X = np.array([X[idx[i]] for i in range(N)])
        y = np.array([y[idx[i]] for i in range(N)])

        f = 'Epochs = {:4d}       Loss = {:8.5f}'
        print('w의 초깃값 ', end='')
        self.printW()
        for j in range(epochs):
            for i in range(N):
                delta = self.predict([X[i]])[0] - y[i]
                self.w -= eta * delta * X[i]
                self.b -= eta * delta
            if j < 10 or (j + 1) % 100 == 0:
                loss = self.predict(X) - y
                loss = (loss * loss).sum() / N
                print(f.format(j + 1, loss), end=' ')
                self.printW()

def visualize(net, X, y, multi_class, labels, class_id, colors, xlabel, ylabel, legend_loc='lower right'):
    x_max = np.ceil(np.max(X[:, 0])).astype(int)
    x_min = np.floor(np.min(X[:, 0])).astype(int)
    y_max = np.ceil(np.max(X[:, 1])).astype(int)
    y_min = np.floor(np.min(X[:, 1])).astype(int)
    x_lin = np.linspace(x_min, x_max, (x_max - x_min) * 20 + 1)
    y_lin = np.linspace(y_min, y_max, (y_max - y_min) * 20 + 1)

    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

    X_test = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    if multi_class:
        y_hat = net.predict(X_test)
        y_hat = np.array([np.argmax(y_hat[k]) for k in range(len(y_hat))], dtype=int)
    else:
        y_hat = (net.predict(X_test) >= 0.5).astype(int)
        y_hat = y_hat.reshape(len(y_hat))

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for c, i, c_name in zip(colors, labels, class_id):
        plt.scatter(X_test[y_hat == i, 0], X_test[y_hat == i, 1],
                    c=c, s=5, alpha=0.3, edgecolors='none')
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    c=c, s=20, label=c_name)

    plt.legend(loc=legend_loc)
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.show()

if __name__ == "__main__":
    nSamples = 200
    nDim = 2

    X_tr, y_tr, labels = prepare_data(nSamples)

    p = Perceptron(nDim, activation=step)
    p.fit(X_tr, y_tr, nSamples, epochs=1000, eta=0.01)

    visualize(p, X_tr, y_tr,
              multi_class=False,
              class_id=labels,
              labels=[0, 1],
              colors=['blue', 'red'],
              xlabel='X',
              ylabel='Y',
              legend_loc='upper left')