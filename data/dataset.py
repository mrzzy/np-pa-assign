#
# NP PA Assignment
# Dataset
#

import os
import numpy as np

# Dataset Columns
cat_features = [
    "Mjob",
    "Fjob",
    "guardian",
    "reason"
]

bin_features = [
    "sex",
    "school",
    "address",
    "Pstatus",
    "famsize",
    "schoolsup",
    "famsup",
    "activities",
    "paid",
    "internet",
    "nursery",
    "higher",
    "romantic",
    "subject"
]

num_features = [
    "age",
    "Medu",
    "Fedu",
    "famrel",
    "traveltime",
    "studytime",
    "failures",
    "freetime",
    "goout",
    "Walc",
    "Dalc",
    "health",
    "absences",
    "G1",
    "G2"
]

input_features = cat_features + bin_features + num_features

target_var = "G3"