# Boston Housing Web App


## Table of Contents
* [Introduction](#introduction)
* [Boston Data Review](#boston-data-review)
* [Code Exploration](#code-exploration)

## Introduction
The housing market is a volatile and there are many variables that could affect the price of a home.

This project aims to predict the median value of the homes using the various variables in the dataset.
## Boston Data Review
The dataset used for this project was sourced from [CMU](http://lib.stat.cmu.edu/datasets/boston), the dataset contains the housing values of suburbs in Boston during the 1970s. 

| Variable      | Description           | 
| ------------- |:---------------------| 
| `CRIM`     | Per capita crime rate per town    |
| `ZN`     | Proportion of residential land zoned for lots over 25,000 sq. ft          |   
| `INDUS` | Proportion of non-retail business acres per town                                         |
| `CHAS`  | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)                                   |
| `NOX`  | Nitric oxides concentration (parts per 10 million)                              |
| `RM`  |  Average number of rooms per dwelling                                     |
| `AGE`  | Proportion of owner-occupied units built prior to 1940                                      |
| `DIS`  | Weighted distances to five Boston employment centres                           |
| `RAD`  | Index of accessibility to radial highways                                       |
| `TAX`  | Full-value property-tax rate per $10,000                                        |
| `PT`  | Pupil-teacher ratio by town                                      |
| `B`  | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town                                         |
| `LSTAT`  | % lower status of the population                            |
| `MEDV`  | Median value of owner-occupied homes in $1000's                                      |

The dataset contains 13 variables and 6072 observations.

## Code Exploration

**To view the full python code for this project, [click here](https://github.com/jidafan/Boston-Housing-Prices-Web-App/blob/main/boston-house-ml-app.py).**

**Importing Relevant Libraries**

```python
import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
```

**Creating a title on the webapp**

```python
st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')
```
![image](https://github.com/jidafan/Boston-Housing-Prices-Web-App/assets/141703009/979ab603-2c68-438f-b1c1-bb15bf49de9e)

**Loading in the data and creating the target array**

```python
data_url = "http://lib.stat.cmu.edu/datasets/boston" 
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) 
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) 
data = np.delete(data,11,1)
target = raw_df.values[1::2, 2]
```
The target array is created from the MEDV column in the dataset, and this is what the program will be trying to predict using the other variables in the dataset.

**Creating a dataframe of the variables in the dataset**

```python
features = pd.read_csv(data_url, nrows = 15, skiprows=range(1,7), header= None)
features = features.drop([12])
features = features.iloc[1:]
features_name = []
for i in features[0]:
    parts = re.split(' ',i.strip())
    features_name.append(parts[0])
```

**Creating X,Y dataframes to be used for our regression model**

```python
X = pd.DataFrame(data, columns=features_name[:-1])
Y = pd.DataFrame(target, columns=["MEDV"])
```
These two dataframes will be used in our RandomForest regression model later.

**Creating a sidebar in the webapp**

```python
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('Crime Rate Per Capita', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('Residental Land Zoned For Lots', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('Non-Retail Business Acres', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('Charles River dummy variable', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('Nitric Oxide Concentration', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('Average number of rooms per dwelling', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('Proportion of owned units prior to 1940', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('Weighted Distance to 5 Boston Employment centres', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('Accesibility to radial highways', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider('Pupil-Teacher Ratio', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    LSTAT = st.sidebar.slider('%Lower Status of population', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
```

![image](https://github.com/jidafan/Boston-Housing-Prices-Web-App/assets/141703009/20b578e2-3faf-48ef-8058-0b38768c7f60)

The sidebar allows users to adjust the values of each variable, the default value is the mean. Adjusting the values changes the graphs on the web app and the predicted median value.

**Creating our predictions**
```python
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')
```

![image](https://github.com/jidafan/Boston-Housing-Prices-Web-App/assets/141703009/945c85bf-7222-4d17-b08f-007cd2881fba)

Creates our prediction of our MEDV and displays it on the webapp

**Creating TreeExplainer Graph**
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')
```

![image](https://github.com/jidafan/Boston-Housing-Prices-Web-App/assets/141703009/981ab7de-de74-4c66-9085-197bd5ac4362)

Displays the importance of each variable to predicting the MEDV in a tree shape. A heatmap is present on the side to showcase how important the feature is. Adjusts based on the values inputted by the user

**Creating Bar Chart**
```python
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
```

![image](https://github.com/jidafan/Boston-Housing-Prices-Web-App/assets/141703009/7f67561f-06b4-43f3-968f-c82f8f9dc573)

Displays the importance of each variable to predicting the MEDV in a bar chart. Adjusts based on the values inputted by the user





