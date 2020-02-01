#
# NP PA Assignment 
# Prepared dataset
#

import os
import joblib
import numpy as np

from . import dataset

# Prepared Dataset Columns
cat_features = dataset.cat_features[:]
bin_features = dataset.bin_features[:] + ["subject"]
num_features = dataset.num_features[:] + [
    "logAbsences", "absencesBinned", "alc",
    "freetimeAlc", "gooutAlc", "goFriendAlc", 
    "failSqrt", "Pedu", "Psup", "Pedusup"
]

input_features = cat_features + bin_features + num_features

# target variables
classify_target_var = "G3Binned"
reg_target_var = "G3"

# Utilities to load preprocessed data
# load & return preprocessed data with the given keys
# keys - list of keys defining which part of the preprocessed data to load
def load_data(keys):
    data =  np.load(os.path.join("build", "dataprep.npz"))
    return [data[key] for key in  keys]

REGRESSION_DATA = ["reg_train_data", "reg_train_scores"]
CLASSIFICATION_DATA = ["classify_train_data", "classify_train_grades" ]
TEST_DATA = ["test_data", "test_scores", "test_grades"]

# load & return objects used to preprocess data
# keys - list of keys defining which objects to load 
def load_objs(keys):
    objects = joblib.load(os.path.join("build", "dataprep.joblib"))
    return [objects[key] for key in keys]