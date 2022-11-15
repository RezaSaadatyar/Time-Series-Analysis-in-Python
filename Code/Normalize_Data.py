import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def normalize_data(Data, Type_Normalize, Display_Figure):
    data = Data.values
    if Data.ndim == 1:
        data = data.reshape(-1, 1)

    if Type_Normalize == 'MinMaxScaler':
        normalize = preprocessing.MinMaxScaler(feature_range=(0, 1))
        normalize.fit(data)
        min_data = normalize.data_min_
        max_data = normalize.data_max_
        normalized_data = normalize.transform(data)  # (Data-min)/(max-min)
    elif Type_Normalize == 'normalize':
        normalized_data = preprocessing.normalize(data, norm='l1', axis=0)  # l1, l2
    if Data.ndim == 1:
        normalized_data = pd.Series(normalized_data.ravel())
    if Display_Figure == 'on':
        plt.rcParams.update({'font.size': 11})
        if Data.ndim == 1:
            plt.subplot(211)
            plt.plot(Data, label='Raw Data'), plt.legend()
            plt.subplot(212)
            plt.plot(normalized_data, label='Normalized Data')
        else:
            plt.subplot(121)
            plt.plot(data[:, 0], data[:, 1], '.', label='Raw Data'), plt.legend()
            plt.subplot(122)
            plt.plot(normalized_data[:, 0], normalized_data[:, 1], '.', label='Normalized Data')
        plt.legend(), plt.tight_layout(), plt.style.use('ggplot'), plt.show()
    return normalized_data, normalize
