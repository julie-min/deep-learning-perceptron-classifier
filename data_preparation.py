import numpy as np

def prepare_data(N):
    C = np.array([[1, 2], [2, 1], [1, 1], [2, 2]])
    X = []
    y = []
    rnd = np.random.default_rng()
    for i in range(N):
        j = rnd.integers(0, 4)
        x = np.array(rnd.normal(loc=0.0, scale=0.2, size=2)) + C[j]
        X.append(x)
        y.append(j // 2)
    return np.array(X), np.array(y), ['0', '1']