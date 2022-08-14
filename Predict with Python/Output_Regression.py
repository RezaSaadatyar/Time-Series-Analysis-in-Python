import numpy as np
import pandas as pd


def output_regression(X_train, X_test, mod, label_train):
    mod.fit(X_train, label_train)
    y_train_pred = pd.Series(mod.predict(X_train))
    y_test_pred = pd.Series(mod.predict(X_test))
    y_test_pred.index = np.arange(X_test.index[0], X_test.index[-1]+1, 1, dtype='int')

    return y_train_pred, y_test_pred
