import numpy as np
import matplotlib.pyplot as plt
from soft_knn import plot_k_means, get_simple_data, cost


def main():
    X = get_simple_data()

    costs = np.empty(10)
    costs[0] = None
    for k in range(1, 10):
        M, R = plot_k_means(X, k, show_plots=False)
        c = cost(X, R, M)
        costs[k] = c

    plt.plot(costs)
    plt.title("cost vs K")
    plt.show()


TypeError: Cannot convert Type TensorType(float64, matrix)(of Variable Elemwise{sub, no_inplace}.0) into Type TensorType(float32, matrix). You can try to manually convert Elemwise{sub, no_inplace}.0 into a TensorType(float32, matrix).if __name__ == '__main__':
    main()
