import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def normalize_regression(data, type_normalize, display_figure):
    data1 = data.values
    if data.ndim == 1:
        data1 = data1.reshape(-1, 1)

    if type_normalize == 'MinMaxScaler':
        normalize_model = preprocessing.MinMaxScaler(feature_range=(0, 1))
        normalize_model.fit(data1)
        min_data = normalize_model.data_min_
        max_data = normalize_model.data_max_
        normalized_data = normalize_model.transform(data1)  # (Data-min)/(max-min)
    elif type_normalize == 'normalize':
        normalized_data = preprocessing.normalize(data1, norm='l1', axis=0)  # l1, l2
    if data.ndim == 1:
        normalized_data = pd.Series(normalized_data.ravel())
        Label = 'Raw Signal'
    else:
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
        Label = data.columns
    if display_figure == 'on':
        plt.rcParams.update({'font.size': 11})
        plt.subplot(211)
        plt.plot(data, label=Label), plt.legend(fontsize=14, ncol=2, frameon=False, loc='best', labelcolor='linecolor', handlelength=0)
        plt.subplot(212)
        plt.plot(normalized_data)
        plt.tight_layout(), plt.style.use('ggplot'), plt.show()
    return normalized_data, normalize_model
