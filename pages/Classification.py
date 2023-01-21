import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from collections import Counter


# some functions
def lower_n_first(string, n):
    string = list(string)
    for i in range(n): string[i] = string[i].lower()
    return "".join(string)



page_names_to_funcs = {
    "Kmeans": kmeans_page,
    # "Fuzzy Kmeans": decision_tree_page,
    "Decision Tree & Random Forest Classifier": tree_page,
    "Logistic Regression": logistic_regression_page,
    "Neural network": neural_network,
}

selected_page = st.sidebar.selectbox("Algorithm for classification", page_names_to_funcs.keys(), key="1")
page_names_to_funcs[selected_page]()