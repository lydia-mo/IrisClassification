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

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

st.write("This line was added after deblpoyement (with a push)")
st.write("## Iris dataset:")

st.write("### 1. Data overview")
#Load data
iris = datasets.load_iris(as_frame=True)
data_target=pd.concat([iris.data, iris.target], axis=1)
data = iris.data
st.dataframe(data_target)

st.session_state['iris'] = datasets.load_iris(as_frame=True)
st.session_state['data_target'] = pd.concat([iris.data, iris.target], axis=1)
st.session_state['data'] = iris.data

