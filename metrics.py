#
# NP PA Assignment
# Additional Metrics 
#

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

## regression
# compute & return the root mean squared error (RMSE)
# y_true - true target values
# y_pred - predicted target values
def root_mean_squared_error(y_true, y_pred, *args, **kwargs):
    return np.sqrt(mean_squared_error(y_true, y_pred, *args, **kwargs))
