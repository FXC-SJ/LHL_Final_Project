import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# streamlit run mlbearings-app.py

st.title('Lighthouse Labs Final Project Bearing Fault Classification')

st.write("""
This app is for Bearing Classification using CNN, ANN, KNN, SVM, RandomForestClassifier

Data obtained from the [CWRU Bearing Data Set](https://engineering.case.edu/bearingdatacenter/download-data-file)
""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://engineering.case.edu/bearingdatacenter/download-data-file)
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine dataset', '0hp - 48 KHz', '1hp - 48 KHz', '2hp - 48 KHz', '3hp - 48 KHz'))

classifier_name = st.sidebar.selectbox('Select Classifier', ('CNN', 'ANN','KNN', 'SVM', 'Random Forest'))

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Bearings Brand',('SKF','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Type',('Ball','Linear'))
        bill_length_mm = st.sidebar.slider('Bearing length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bearing depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Housing length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Bearing mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset', X.shape)
st.write('Number of classes', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f'classifier = {classifier_name}')
st.write(f'accuracy = {acc}')

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)

# TODO
#- add more parameters (sklearn)
#- add more classifier
#- add feature scaling















