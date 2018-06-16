import numpy as np
import pandas as pd


def get_data(file_name):
    df = pd.read_csv(file_name)
    # print(df.head())

    df = df.as_matrix()
    X = df[:, :-1]
    Y = df[:, -1:]

    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    print(X2)

    # simple way
    for n in range(N):
        # ONE HOT ENCODEING FOR THE COLOUMNS
        t = int(X[n, D - 1])  # time of the day przyjmuje wartosci [0, 1, 2, 3]
        X2[n, t + D - 1] = 1  # w zaleznosci od tego ktora to wartosc to dodajemy do odpowieniej kolumny 1

    print(X2)

    Z = np.zeros((N, 4))  # just create matrix with 4 columns
    Z[np.arange(N), X[:, D - 1].astype(np.int32)] = 1  # 1 dim is just 1 to N and 2nd i pass categories ### just Z[all, right column] get value 1
    # X2[:,-4:] = Z # put values into the matrix
    #print(X[:, D - 1].astype(np.int32))
    # print(Z)
    print([np.arange(N), X[:, D - 1].astype(np.int32)])

    # if its true its ok if its false python rasises the assertion error
    assert(np.abs(X2[:, -4:] - Z).sum() < 10e-10)

    return X2, Y


def get_binary_data():
    X, Y = get_data('ecommerce_data.csv')
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


#X, Y = get_data('ecommerce_data.csv')
