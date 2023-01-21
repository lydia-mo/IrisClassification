import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import datasets

from collections import Counter

st.write("""
# Simple Iris Flower Prediction App

This app allows to predict the **Iris flower** type by **user** input values\n
It uses different machine learning algorithms to achieve that prediction :
""")
st.markdown("- Kmeans")
st.markdown("- Fuzzy Kmeans")
st.markdown("- Decision Tree")
st.markdown("- Random Forest Classifier")
st.markdown("- Logistic Regression")
st.markdown("- Neural network")