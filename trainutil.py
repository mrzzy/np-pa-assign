#
# NP PA assignment
# Model Training Utils
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import base
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.feature_selection import RFECV

# evaluate models with given metrics, data using K fold cross validation
# models -list of tuple of (model_name, model)
# metrics - list of tuple of (name, metric, greater is better)
# data - tuple of training input data and target outputs
# cv - no. of cross validation folds
# returns dataframe with evaluation results
def evaluate_models(models, metrics, data, cv):
    # evaluate models with metrics
    results = {}
    for model_name, model in models:
        X, y = data
        scorer_dict = {name: make_scorer(metric, greater_is_better) 
                       for name, metric, greater_is_better in metrics}
        results[model_name] = cross_validate(model, X, y,
                                             scoring=scorer_dict,
                                             cv=cv, n_jobs=-1)

    # construct dataframe with evaluation results
    model_names = [name for name, _ in models ]
    metrics_metadata = [(name, greater_is_better) for name, _, greater_is_better  in metrics]

    # compile columns of the dataframes
    results_cols = {}
    results_cols["model_names"] = np.concatenate([[name] * cv for name in model_names])
    results_cols["eval_times"] =  np.concatenate([result["score_time"] for result in list(results.values())])
    # compile results from each metric
    for model_name in model_names:
        for col, greater_is_better in metrics_metadata:
            # create list for col if does already exist
            if not col in results_cols: results_cols[col] = []
            # append results into respective columns 
            result = results[model_name][f"test_{col}"]
            results_cols[col].append(result if greater_is_better else -result)
    for col, _ in metrics_metadata:
        results_cols[col] = np.concatenate(results_cols[col])
        
    results_df = pd.DataFrame.from_dict(results_cols)
    return results_df


# plot evaluation results of the given results df
# n_col - no. of plots to fit per in each row 
# figsize - size of the plot
# ci - percentage confidence of confidence interval drawn in plot
def plot_eval_results(results_df, n_col=3, figsize=(8, 8), ci=95):
    metric_names = [ col for col in results_df.columns if col != "model_names"]
    n_row = len(metric_names) // n_col 
    if len(metric_names) % n_col > 0.0:
        n_row += 1
    fig = plt.figure(figsize=figsize)
    for i, metric_name in enumerate(metric_names): 
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.set_title(metric_name)
        sns.barplot(metric_name, "model_names", 
                    data=results_df, ci=ci)


    fig.tight_layout()
    fig.show()

# plot a learning curve of the models using given training data
# models - tuple of (name, model) of the models, each producing a learning curve
# metric - metric used to evaluate the model
# n_cols - no. of columns used in plot
# data - tuple of training data (input_data, target_outputs)
# train_size - fractions of the training set size to use to validate
# cv - no. of cross validation splits used to estimate the validation curve
def plot_learning_curve(models, metric, n_col, data,
                        figsize=(10, 10), ylim=None,
                        train_sizes=np.linspace(0.1, 1.0, 8), cv=12):
    X, y = data
    scorer = make_scorer(metric)

    # calculate n rows and cols in plot
    n_row = len(models) // n_col 
    if len(models) % n_col > 0.0:
        n_row += 1
    
    fig = plt.figure(figsize=figsize)
    for i, m in enumerate(models):
        name, model = m
        ax = fig.add_subplot(n_row, n_col, i + 1)
        # evaluate model with varying training sizes
        train_size, train_metric_vals, valid_metric_vals = learning_curve(model,  X,  y, 
                                                                          n_jobs=-1, cv=cv,
                                                                          scoring=scorer,
                                                                          train_sizes=train_sizes)

        # average across validation fold axis to get average metric
        # value for each training size
        train_metric_vals = np.mean(train_metric_vals, axis=-1)
        valid_metric_vals = np.mean(valid_metric_vals, axis=-1)
        # fix axis
        if not ylim is None: plt.ylim(*ylim)

        ax.set_title(name)
        sns.lineplot(train_size, train_metric_vals, label="train")
        ax.plot(train_size, valid_metric_vals, label="valid")
        ax.legend()
    fig.tight_layout()
    fig.show()