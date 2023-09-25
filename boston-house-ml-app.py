import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston" 
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) 
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) 
data = np.delete(data,11,1)
target = raw_df.values[1::2, 2]
features = pd.read_csv(data_url, nrows = 15, skiprows=range(1,7), header= None)
features = features.drop([12])
features = features.iloc[1:]
features_name = []
for i in features[0]:
    parts = re.split(' ',i.strip())
    features_name.append(parts[0])
X = pd.DataFrame(data, columns=features_name[:-1])
Y = pd.DataFrame(target, columns=["MEDV"])

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

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
