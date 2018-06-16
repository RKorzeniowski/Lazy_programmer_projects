import numpy as np
import pandas as pd

# df.head()


def get_data():

    df = pd.read_csv('ecommerce_data.csv')
    data. df.as_matrix()

    X = data[:, : -1]
    Y = data[:, -1]

    # normalize
    X = data[:, : -1]
