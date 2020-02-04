#
# NP PA Assignment 
# Prepared dataset
#

import os
import joblib
import pandas as pd
import numpy as np
from sklearn import base

from . import dataset

# Prepared Dataset Columns
cat_features = [
    "Mjob",
    "Fjob",
    "guardian",
    "reason"
]

bin_features = [
    "school",
    "address",
    "famsize",
    "schoolsup",
    "activities",
    "paid",
    "internet",
    "nursery",
    "higher",
    "romantic",
    "subject"
]

num_features = [
    "G1",
    "G2",
    "goFreeAlcAge",
    "alcStudyAge",
    "locVal",
    "agePow",
    "failSqrt",
]

input_features = cat_features + bin_features + num_features

# target variables
classify_target_var = "G3Binned"
reg_target_var = "G3"

# Utilities to load preprocessed data
# load & return preprocessed data with the specifiers:
# tasks - intended task of data: "classify" or "regression"
# subjects - "math" or "portguese"
# subsets - train or test subset
# returns list of input_data, output target for subset 1, subset 2 and so on.
def load_data(task, subject, subsets=["train", "test"]):
    dump = joblib.load(os.path.join("build", "dataprep.joblib"))
    input_data, outputs = dump["input_data"], dump["outputs"]
    returns = []
    for subset in subsets:
        subset_id = f"{subject}_{subset}"
        X, y = input_data[task][subset_id], outputs[task][subset_id]
        returns += [X, y]
    return returns

# load obj from preprocessed data specified by the given key
# key - key used to select the preprocesed data to load
def load_key(key):
    dump = joblib.load(os.path.join("build", "dataprep.joblib"))
    return dump[key]

# feature extractor use to extract featured discovered in
# feature enginnering
class FeatureExtractor(base.TransformerMixin):
    def fit(self, X, y=None):
        return self # nothing to do
    # transform op adds features engineered to given dataframe 
    # df - dataframe to add features to
    # y - not used 
    def transform(self, df, y=None):
        df = df.copy()
        # vices
        df["alc"]= df["Dalc"] + df["Walc"]
        df["goFreeAlcAge"] = df["goout"] * df["freetime"] * df["alc"] * df["age"] 
        df["alcStudyAge"] = df["alc"] * df["age"] / df["studytime"] 
        
        # negative factors
        df["failSqrt"] = np.sqrt(df["failures"]) 
        df["agePow"] = df["age"] ** 6.5
        
        # family/location
        df["locVal"] = df["Medu"] / df["traveltime"]
        
        return df