# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
  predicted = model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
  if predicted[0]==1:
    return 'building windows float processed'
  elif predicted[0]==2:
    return 'building windows non float processed'
  elif predicted[0]==3:
    return 'vehicle windows float processed'
  elif predicted[0]==4:
    return 'vehicle windows non float processed'
  elif predicted[0]==5:
    return 'containers'
  elif predicted[0]==6:
    return 'tableware'
  elif predicted[0]==7:
    return 'headlamps'

st.title('GLASSTYPE PREDICTOR')
st.sidebar.title('Exploratory Data Analysis')
if st.sidebar.checkbox('Show row Data'):
  st.subheader('GLASSTYPE DATASET')
  st.dataframe(glass_df)

st.set_option('deprecation.showPyplotGlobalUse',False)
st.sidebar.subheader('Scatter Plot')
features_scat = st.sidebar.multiselect('Select the x-axis values:',tuple(glass_df.columns[:-1]))
for i in features_scat:
  st.subheader(f'Scatter plot between {i} and glasstype')
  plt.figure(figsize=(20,5))
  plt.scatter(glass_df[i],glass_df['GlassType'])
  st.pyplot()


st.sidebar.subheader('Histogram')
features_hist = st.sidebar.multiselect('Select the features to create histogram:',tuple(glass_df.columns[:-1]))
for i in features_hist:
  st.subheader(f'Histogram for {i}')
  plt.figure(figsize=(20,5))
  plt.hist(glass_df[i],bins='sturges')
  st.pyplot()


st.sidebar.subheader('Boxplot')
features_box = st.sidebar.multiselect('Select the features to create boxplot:',tuple(glass_df.columns[:-1]))
for i in features_box:
  st.subheader(f'Boxplot for {i}')
  plt.figure(figsize=(20,5))
  sns.boxplot(x='GlassType',y=i,data=glass_df)
  st.pyplot()
