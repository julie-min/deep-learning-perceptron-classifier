from data_preparation import prepare_data
from single_layer_perceptron import Perceptron, step, visualize as slp_visualize
from multi_layer_perceptron import MLPClassifier, train_test_split, visualize as mlp_visualize

def run_single_layer_perceptron():
    print("Running Single Layer Perceptron...")
    nSamples = 200
    nDim = 2
    X_tr, y_tr, labels = prepare_data(nSamples)
    p = Perceptron(nDim, activation=step)
    p.fit(X_tr, y_tr, nSamples, epochs=1000, eta=0.01)
    slp_visualize(p, X_tr, y_tr,
                  multi_class=False,
                  class_id=labels,
                  labels=[0, 1],
                  colors=['blue', 'red'],
                  xlabel='X',
                  ylabel='Y',
                  legend_loc='upper left')

def run_multi_layer_perceptron():
    print("Running Multi-Layer Perceptron...")
    nSamples = 1000
    X, y, labels = prepare_data(nSamples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    print(f"Training accuracy: {mlp.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {mlp.score(X_test, y_test):.4f}")
    mlp_visualize(mlp, X, y, labels, ['blue', 'red'], 'X', 'Y')

if __name__ == "__main__":
    run_single_layer_perceptron()
    run_multi_layer_perceptron()