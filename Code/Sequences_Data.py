import numpy as np


def sequences_data(dataset, seq_size=1):
    """
    Args:
        dataset: Creates a dataset where x is the number of values at a given time (t, t-1, ...) and y is the number
         of values at the next time (t+1)
        seq_size: It is the number of previous time steps to use as input variables to predict the next time period.
        Larger sequences (look further back) may improve forecasting.
    Returns: Variables for train and test LSTM
    """
    x = []
    y = []
    dataset = dataset.reshape(-1, 1)
    for k in range(len(dataset) - seq_size):
        window = dataset[k:(k + seq_size), 0]
        x.append(window)
        y.append(dataset[k + seq_size, 0])
    x = np.reshape(np.array(x), (np.array(x).shape[0], 1, np.array(x).shape[1]))  # Reshape input to be [samples, time steps, features]
    return x, np.array(y)
