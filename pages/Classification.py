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

def kmeans_page():
    st.markdown("# KMeans Clustering")
    st.write("Applying KMeans (from sklearn) on our dataset")

    try:
        data = st.session_state["data"]
    except Exception as e:
        switch_page("Home")
    inertia_lst = [KMeans(n_clusters=i, random_state=0, n_init="auto").fit(data).inertia_ for i in range(1, 11)]
    fig= plt.figure(figsize=(10, 4))
    sns.scatterplot(x=range(1, 11), y=inertia_lst)
    st.pyplot(fig)

    nb_clusters = 3
    try:
        nb_clusters  = int(st.text_input('Number of clusters', 
                            placeholder="Specify the number of clusters"))
    except Exception as e :
        print(e)
    st.write('## The current number of clusters is', nb_clusters)

    # Classification based on the number of cluster specified
    model_kmeans = KMeans(n_clusters=nb_clusters, random_state=0, n_init="auto").fit(data)
    st.session_state["model_kmeans"] = model_kmeans
    c_labels = model_kmeans.labels_
    counter = Counter(c_labels)
    fig= plt.figure(figsize=(10, 4))
    sns.barplot(x=list(counter.keys()), y=list(counter.values())).set(title='Distrubition of flowers classes')
    del counter
    st.write("### Distibution of classes based on the Kmeans results")
    st.pyplot(fig)

    #Visualisation
    st.write("### Visualisation of data with PCA dimension reduction")
    option = st.selectbox(
        'Select the number of dimensions:',
        ('2D', '3D'),)
    st.write('#### %s plot :' %option)
    if option=='2D':
        data_pca_2d=st.session_state['data_pca_2d']
        x, y = data_pca_2d.T
        fig, axes= plt.subplots(1, 2, figsize=(9,9), sharex=True)
        axes[0].set_title('Original labels')
        axes[1].set_title('New labels')
        sns.set(style='whitegrid')
        sns.scatterplot(x=x, y=y, c=st.session_state["iris"].target, ax=axes[0])
        sns.scatterplot(x=x, y=y, c=c_labels, ax=axes[1])
        st.pyplot(fig)
    else:
        data_pca_3d=st.session_state['data_pca_3d']
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(*data_pca_3d.T, c=st.session_state["iris"].target, marker='o', depthshade=False, cmap='Paired')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("Original labels")
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(*data_pca_3d.T, c=c_labels, marker='o', depthshade=False, cmap='Paired')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("New labels")
        st.pyplot(fig)

    if nb_clusters==3:
        accuracy_kmeans = 1 - np.sum(np.abs(np.array(list(Counter(c_labels).values())) - 50))/len(c_labels)
        st.write("### Approximated accurary:", accuracy_kmeans)


page_names_to_funcs = {
    "Kmeans": kmeans_page,
    # "Fuzzy Kmeans": decision_tree_page,
    "Decision Tree & Random Forest Classifier": tree_page,
    "Logistic Regression": logistic_regression_page,
    "Neural network": neural_network,
}

selected_page = st.sidebar.selectbox("Algorithm for classification", page_names_to_funcs.keys(), key="1")
page_names_to_funcs[selected_page]()