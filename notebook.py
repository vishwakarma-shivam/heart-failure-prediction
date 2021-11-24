import streamlit as st

nb = '''
# Notebook
### Table of Content   
* [Data](#0)
* [What Problem We Have and Which Metric to Use?](#1)

* [Exploratory Data Analysis](#2)
    * [Target Variable](#3)
    * [Numerical Features](#4)
    * [Categorical Features](#5)    
    
* [Model Selection](#6)    
    * [Baseline Model](#7)
    * [Logistic & Linear Discriminant & SVC & KNN](#8)
    * [Logistic & Linear Discriminant & SVC & KNN with Scaler](#9)    
    * [Ensemble Models (AdaBoost & Gradient Boosting & Random Forest & Extra Trees)](#10)
    * [Famous Trio (XGBoost & LightGBM & Catboost)](#11)
    * [CATBOOST](#12)
    * [Catboost HyperParameter Tuning with OPTUNA](#13)
    * [Feature Importance](#14)    
    * [Model Comparison](#15)  
    
    


* [Conclusion](#16)

* [References & Further Reading](#17)

<a id="0"></a>
<font color="lightseagreen" size=+2.5><b>Data</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# Heart Failure Prediction Dataset

DATA DICTONARY						
						
1	**Age**: 			Age of the patient [years] 		
2	**Sex**:  			 Sex of the patient [M: Male, F: Female] 		
3	**ChestPainType**: 			[TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic] 		
4	**RestingBP**:			Resting blood pressure [mm Hg] 		
5	**Cholesterol**:			Serum cholesterol [mm/dl] 		
6	**FastingBS**:			 Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]		
7	**RestingECG**:			 Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria] 		
8	**MaxHR**:			Maximum heart rate achieved [Numeric value between 60 and 202]		
9	**ExerciseAngina**:			Exercise-induced angina [Y: Yes, N: No]		
10	**Oldpeak**:			 ST [Numeric value measured in depression] (		
11	**ST_Slope**:			 The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping] 		
12	**HeartDisease**:			 Output class [1: heart disease, 0: Normal] 		

Reference: https://www.kaggle.com/fedesoriano/heart-failure-prediction

<a id="1"></a>
<font color="lightseagreen" size=+1.5><b>What Problem We Have and Which Metric to Use?</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

- Based on the data and data dictionary, We have a classification problem.
- We wil make classification on the target variable **Heart Disease**
- And we will build a model to get best calssification possible on the target variable.
- For that we will look at the balance of the target variable.
- As we will see later, our target variable has balanced or balanced like data
- For that reason we will use **Accuracy score**.
- [For the detailed info about the evaluation metrics](https://www.kaggle.com/kaanboke/the-most-common-evaluation-metrics-a-gentle-intro)

<a id="2"></a>
<font color="lightseagreen" size=+2.5><b>Exploratory Data Analysis</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

- Let's import the libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




```


```python
from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,classification_report

#importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import missingno as msno

import warnings
warnings.filterwarnings("ignore")
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.6.3.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
pd.set_option('max_columns',100)
pd.set_option('max_rows',900)

pd.set_option('max_colwidth',200)

df = pd.read_csv('heart.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>108</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 918 entries, 0 to 917
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Age             918 non-null    int64  
     1   Sex             918 non-null    object 
     2   ChestPainType   918 non-null    object 
     3   RestingBP       918 non-null    int64  
     4   Cholesterol     918 non-null    int64  
     5   FastingBS       918 non-null    int64  
     6   RestingECG      918 non-null    object 
     7   MaxHR           918 non-null    int64  
     8   ExerciseAngina  918 non-null    object 
     9   Oldpeak         918 non-null    float64
     10  ST_Slope        918 non-null    object 
     11  HeartDisease    918 non-null    int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 86.2+ KB
    

- Overall data types seems ok.


```python
df.duplicated().sum()
```




    0



- No duplicates


```python
def missing (df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values

missing(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing_Number</th>
      <th>Missing_Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ChestPainType</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RestingBP</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>FastingBS</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RestingECG</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>MaxHR</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ExerciseAngina</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Oldpeak</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ST_Slope</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>HeartDisease</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



- No missing values.


```python
numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')
```

    Numerical Columns:  Index(['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], dtype='object')
    
    
    Categorical Columns: Index(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], dtype='object')
    


```python
df[categorical].nunique()
```




    Sex               2
    ChestPainType     4
    RestingECG        3
    ExerciseAngina    2
    ST_Slope          3
    dtype: int64



- So far so good. No zero variance and no extremely high variance.

<a id="3"></a>
<font color="lightseagreen" size=+1.5><b>Target Variable</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
y = df['HeartDisease']
print(f'Percentage of patient had a HeartDisease:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} patient)\nPercentage of patient did not have a HeartDisease: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} patient)')
```

    Percentage of patient had a HeartDisease:  55.34 %  --> (508 patient)
    Percentage of patient did not have a HeartDisease: 44.66  %  --> (410 patient)
    

- Almost 55% of the patients had a heart disease.
-  508 patient had a heart disease.
- Almost 45%  of patients didn't have a heart disease.
- 410 patient didn't have a heart disease.



```python
df['HeartDisease'].iplot(kind='hist')
```


<div>                            <div id="6947dba5-cefa-46b0-8c6f-de224e9628bd" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';                                    if (document.getElementById("6947dba5-cefa-46b0-8c6f-de224e9628bd")) {                    Plotly.newPlot(                        "6947dba5-cefa-46b0-8c6f-de224e9628bd",                        [{"histfunc":"count","histnorm":"","marker":{"color":"rgba(255, 153, 51, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"HeartDisease","opacity":0.8,"orientation":"v","type":"histogram","x":[0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0]}],                        {"barmode":"overlay","legend":{"bgcolor":"#F5F6F9","font":{"color":"#4D5663"}},"paper_bgcolor":"#F5F6F9","plot_bgcolor":"#F5F6F9","template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"font":{"color":"#4D5663"}},"xaxis":{"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis":{"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"}},                        {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}                    ).then(function(){

var gd = document.getElementById('6947dba5-cefa-46b0-8c6f-de224e9628bd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- There is a little imblanace but nothing in the disturbing level.
- We can use 'accuracy' metric as our evaluation metric.

<a id="4"></a>
<font color="lightseagreen" size=+1.5><b>Numerical Features</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
df[numerical].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>MaxHR</th>
      <th>Oldpeak</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.510893</td>
      <td>132.396514</td>
      <td>198.799564</td>
      <td>0.233115</td>
      <td>136.809368</td>
      <td>0.887364</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.432617</td>
      <td>18.514154</td>
      <td>109.384145</td>
      <td>0.423046</td>
      <td>25.460334</td>
      <td>1.066570</td>
    </tr>
    <tr>
      <th>min</th>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>-2.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.000000</td>
      <td>120.000000</td>
      <td>173.250000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>130.000000</td>
      <td>223.000000</td>
      <td>0.000000</td>
      <td>138.000000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>140.000000</td>
      <td>267.000000</td>
      <td>0.000000</td>
      <td>156.000000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>200.000000</td>
      <td>603.000000</td>
      <td>1.000000</td>
      <td>202.000000</td>
      <td>6.200000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[numerical].iplot(kind='hist');
```


<div>                            <div id="cf355976-6b4c-42ce-930f-6addee9d3ac9" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';                                    if (document.getElementById("cf355976-6b4c-42ce-930f-6addee9d3ac9")) {                    Plotly.newPlot(                        "cf355976-6b4c-42ce-930f-6addee9d3ac9",                        [{"histfunc":"count","histnorm":"","marker":{"color":"rgba(255, 153, 51, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Age","opacity":0.8,"orientation":"v","type":"histogram","x":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38]},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(55, 128, 191, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"RestingBP","opacity":0.8,"orientation":"v","type":"histogram","x":[140,160,130,138,150,120,130,110,140,120,130,136,120,140,115,120,110,120,100,120,100,120,124,150,130,130,124,120,113,125,145,130,125,130,150,125,140,110,120,150,150,130,150,140,120,130,120,140,112,110,130,120,140,130,130,160,120,130,150,112,100,150,140,120,110,120,132,110,160,150,140,130,120,120,140,150,118,140,140,130,110,120,150,160,150,140,170,140,120,140,110,130,120,160,110,130,142,160,120,125,130,130,150,120,118,140,120,150,140,190,130,150,140,140,130,100,120,130,120,140,135,125,110,180,130,120,130,108,120,120,145,110,170,150,130,115,120,120,140,150,160,140,160,140,120,110,120,120,120,130,130,100,130,120,120,155,110,140,130,160,140,128,160,120,140,140,140,140,135,140,120,140,140,140,140,140,140,140,130,130,130,130,140,110,160,160,130,120,120,180,180,170,130,135,125,160,120,150,120,130,110,120,160,100,130,150,120,110,130,125,106,140,130,130,150,170,110,120,140,140,130,160,120,120,120,145,120,92,120,130,130,130,120,112,140,120,120,140,160,160,145,200,160,120,160,120,120,122,130,130,135,120,125,140,145,120,130,150,150,122,140,120,120,130,140,160,130,98,130,130,120,105,140,120,180,180,135,170,180,130,120,150,130,110,140,110,140,120,133,120,110,140,130,115,95,105,145,110,110,110,160,140,125,120,95,120,115,130,115,95,155,125,125,115,80,145,105,140,130,145,125,100,105,115,100,105,110,125,95,130,115,115,100,95,130,120,160,150,140,95,100,110,110,130,120,135,120,115,137,110,120,140,120,130,120,145,115,120,115,105,160,160,155,120,120,200,150,135,140,150,135,150,185,135,125,160,155,160,140,120,160,115,115,110,120,150,145,130,140,160,140,115,130,150,160,135,140,170,165,200,160,130,145,135,110,120,140,115,110,160,150,180,125,125,130,155,140,130,132,142,110,120,150,180,120,160,126,140,110,133,128,120,170,110,126,152,116,120,130,138,128,130,128,130,120,136,130,124,160,0,122,144,140,120,136,154,120,125,134,104,139,136,122,128,131,134,120,132,152,124,126,138,154,141,131,178,132,110,130,170,126,140,142,120,134,139,110,140,140,136,120,170,130,137,142,142,132,146,160,135,136,130,140,132,158,136,136,106,120,110,136,160,123,112,122,130,150,150,102,96,130,120,144,124,150,130,144,139,131,143,133,143,116,110,125,130,133,150,130,110,138,104,138,170,140,132,132,142,112,139,172,120,144,145,155,150,160,137,137,134,133,132,140,135,144,141,150,130,110,158,128,140,150,160,142,137,139,146,156,145,131,140,122,142,141,180,124,118,140,140,136,100,190,130,160,130,122,133,120,130,130,140,120,155,134,114,160,144,158,134,127,135,122,140,120,130,115,124,128,120,120,130,110,140,150,135,142,140,134,128,112,140,140,110,140,120,130,115,112,132,130,138,120,112,110,128,160,120,170,144,130,140,160,130,122,152,124,130,101,126,140,118,110,160,150,136,128,140,140,130,105,138,120,174,120,150,130,120,150,145,150,140,136,118,108,120,120,156,140,106,142,104,94,120,120,146,120,150,130,110,148,128,178,126,150,140,130,124,110,125,110,120,100,140,120,108,120,130,165,130,124,100,150,140,112,180,110,158,135,120,134,120,200,150,130,120,122,152,160,125,160,120,136,134,117,108,112,140,120,150,142,152,125,118,132,145,138,140,125,192,123,112,110,132,112,112,120,108,130,130,105,140,128,120,178,120,150,130,128,110,180,110,130,138,138,160,140,100,120,118,138,140,150,125,129,120,134,110,102,130,130,132,108,140,160,140,145,108,126,124,135,100,110,140,125,118,125,125,140,160,152,102,105,125,130,170,125,122,128,130,130,135,94,120,120,110,135,150,130,138,135,130,132,150,118,145,118,115,128,130,160,138,120,138,120,180,140,130,140,140,130,110,155,140,145,120,130,112,110,150,160,150,132,140,150,120,130,120,130,110,172,120,140,140,160,128,138,132,128,134,170,146,138,154,130,110,130,128,122,148,114,170,125,130,120,152,132,120,140,124,120,164,140,110,144,130,130,138]},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(50, 171, 96, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Cholesterol","opacity":0.8,"orientation":"v","type":"histogram","x":[289,180,283,214,195,339,237,208,207,284,211,164,204,234,211,273,196,201,248,267,223,184,201,288,215,209,260,284,468,188,518,167,224,172,186,254,306,250,177,227,230,294,264,259,175,318,223,216,340,289,233,205,224,245,180,194,270,213,365,342,253,254,224,277,202,260,297,225,246,412,265,215,182,218,268,163,529,167,100,206,277,238,223,196,213,139,263,216,291,229,208,307,210,329,182,263,207,147,85,269,275,179,392,466,186,260,254,214,129,241,188,255,276,297,207,246,282,338,160,156,248,272,240,393,230,246,161,163,230,228,292,202,388,230,294,265,215,241,166,247,331,341,291,243,279,273,198,249,168,603,215,159,275,270,291,342,190,185,290,195,264,212,263,196,225,272,231,238,222,179,243,235,320,187,266,288,216,287,194,238,225,224,404,238,312,211,251,237,328,285,280,209,245,192,184,193,297,268,246,308,249,230,147,219,184,215,308,257,132,216,263,288,276,219,226,237,280,217,196,263,222,303,195,298,256,264,195,117,295,173,315,281,275,250,309,200,336,295,355,193,326,198,292,266,268,171,237,275,219,341,491,260,292,271,248,274,394,160,200,320,275,221,231,126,193,305,298,220,242,235,225,198,201,220,295,213,160,223,347,253,246,222,220,344,358,190,169,181,308,166,211,257,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,260,209,218,228,213,0,236,0,0,267,166,0,0,0,0,220,177,236,0,0,0,0,0,0,0,0,0,186,100,228,0,171,230,0,0,0,281,0,203,0,0,0,0,0,277,0,233,0,0,240,0,0,153,224,0,0,0,316,0,0,218,0,311,0,0,0,270,0,0,217,214,214,252,220,214,203,0,339,216,276,458,241,384,297,248,308,208,227,210,245,225,240,0,198,195,267,161,258,0,0,195,235,0,305,223,282,349,160,160,236,312,283,142,211,218,306,186,252,222,0,0,258,202,197,204,113,274,192,298,272,220,200,261,181,260,220,221,216,175,219,310,208,232,273,203,182,274,204,270,292,171,221,289,217,223,110,193,123,210,282,170,369,173,289,152,208,216,271,244,285,243,240,219,237,165,213,287,258,256,186,264,185,226,203,207,284,337,310,254,258,254,300,170,310,333,139,223,385,254,322,564,261,263,269,177,256,239,293,407,234,226,235,234,303,149,311,203,211,199,229,245,303,204,288,275,243,295,230,265,229,228,215,326,200,256,207,273,180,222,223,209,233,197,218,211,149,197,246,225,315,205,417,195,234,198,166,178,249,281,126,305,226,240,233,276,261,319,242,243,260,354,245,197,223,309,208,199,209,236,218,198,270,214,201,244,208,270,306,243,221,330,266,206,212,275,302,234,313,244,141,237,269,289,254,274,222,258,177,160,327,235,305,304,295,271,249,288,226,283,188,286,274,360,273,201,267,196,201,230,269,212,226,246,232,177,277,249,210,207,212,271,233,213,283,282,230,167,224,268,250,219,267,303,256,204,217,308,193,228,231,244,262,259,211,325,254,197,236,282,234,254,299,211,182,294,298,231,254,196,240,409,172,265,246,315,184,233,394,269,239,174,309,282,255,250,248,214,239,304,277,300,258,299,289,298,318,240,309,250,288,245,213,216,204,204,252,227,258,220,239,254,168,330,183,203,263,341,283,186,307,219,260,255,231,164,234,177,257,325,274,321,264,268,308,253,248,269,185,282,188,219,290,175,212,302,243,353,335,247,340,206,284,266,229,199,263,294,192,286,216,223,247,204,204,227,278,220,232,197,335,253,205,192,203,318,225,220,221,240,212,342,169,187,197,157,176,241,264,193,131,236,175]},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(128, 0, 128, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"FastingBS","opacity":0.8,"orientation":"v","type":"histogram","x":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0]},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(219, 64, 82, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"MaxHR","opacity":0.8,"orientation":"v","type":"histogram","x":[172,156,98,108,122,170,170,142,130,120,142,99,145,140,137,150,166,165,125,160,142,142,164,150,138,178,112,118,127,145,130,114,122,130,154,155,87,142,148,130,130,100,168,170,120,120,168,170,184,170,121,98,122,150,140,170,153,140,134,96,174,175,144,125,145,130,144,184,82,170,145,135,150,115,128,116,130,150,138,170,160,154,115,165,125,94,112,142,155,110,160,140,148,92,180,140,138,160,140,144,115,100,130,152,124,140,110,168,135,106,124,92,125,150,135,150,170,130,185,180,170,139,140,110,150,110,190,175,140,152,130,150,122,124,120,175,175,146,118,130,94,125,158,155,150,132,155,176,160,125,120,100,150,140,160,150,150,130,100,130,119,96,174,150,140,175,140,118,100,160,160,188,162,172,134,135,105,150,150,90,120,150,124,140,130,92,110,138,110,120,120,116,160,110,180,116,132,136,116,98,150,150,146,150,100,140,180,140,185,140,110,140,128,164,98,170,150,137,150,170,112,150,125,185,137,150,140,134,170,184,158,167,129,142,140,160,118,136,99,102,155,142,143,118,103,137,150,150,130,120,135,115,115,152,96,130,150,172,120,155,165,138,115,125,145,175,110,150,91,145,140,165,130,134,180,100,150,126,126,155,135,122,160,160,170,120,140,132,156,180,138,135,148,93,127,110,139,131,92,149,149,150,120,123,126,127,155,120,138,182,154,110,176,154,141,123,148,121,77,136,175,109,166,128,133,128,138,119,82,130,143,82,179,144,170,134,114,154,149,145,122,114,113,120,104,130,115,128,104,125,120,140,100,100,92,125,113,95,128,115,72,124,99,148,97,140,117,120,120,86,63,108,98,115,105,121,118,122,157,156,99,120,145,156,155,105,99,135,83,145,60,92,115,120,98,150,143,105,122,70,110,163,67,128,120,130,100,72,94,122,78,150,103,98,110,90,112,127,140,149,99,120,105,140,141,157,140,117,120,120,148,86,84,125,120,118,124,106,111,116,180,129,125,140,120,124,117,110,105,155,110,122,118,133,123,131,80,165,86,111,118,84,117,107,128,160,125,130,97,161,106,130,140,122,130,120,139,108,148,123,110,118,125,106,112,128,180,144,135,140,102,108,145,127,110,140,69,148,130,130,140,138,140,138,112,131,112,80,150,110,126,88,153,150,120,160,132,120,110,121,128,135,120,117,150,144,113,135,127,109,128,115,102,140,135,122,119,130,112,100,122,120,105,129,120,139,162,100,140,135,73,86,108,116,160,118,112,122,124,102,137,141,154,126,160,115,128,115,105,110,119,109,135,130,112,126,120,110,119,110,130,159,84,126,116,120,122,165,122,94,133,110,150,130,113,140,100,136,127,98,96,123,98,112,151,96,108,128,138,126,154,137,100,135,93,109,160,141,105,121,140,142,142,170,154,161,111,180,145,159,125,120,155,144,178,129,180,181,143,159,139,152,157,165,130,150,138,170,140,126,150,138,125,150,186,181,163,179,156,134,165,126,177,120,114,125,184,157,179,175,168,125,96,143,103,173,142,169,171,150,112,186,152,149,152,140,163,143,116,142,147,148,179,173,178,105,130,111,168,126,178,140,145,163,128,164,169,109,108,168,118,151,156,133,162,175,71,163,124,147,166,143,157,162,138,117,153,161,170,162,162,144,133,114,103,139,116,88,151,152,163,99,169,158,160,169,132,178,96,165,160,172,144,192,168,132,182,163,125,195,95,160,114,173,172,179,158,167,122,149,172,111,170,162,165,182,154,155,130,161,154,159,152,152,174,131,146,125,115,174,106,122,147,163,163,194,150,158,122,173,162,105,147,157,112,160,125,156,156,175,161,122,158,151,162,151,171,141,173,145,178,160,154,131,187,159,166,165,131,202,172,172,154,147,170,126,127,174,132,182,132,97,136,162,190,146,140,185,161,146,145,160,120,156,172,150,182,143,160,142,144,158,148,155,142,113,188,153,123,157,162,137,132,158,171,172,132,160,171,168,162,173,153,148,108,115,169,143,156,162,155,152,152,164,131,143,179,130,174,161,140,146,144,163,169,150,166,144,144,136,182,90,123,132,141,115,174,173]},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(0, 128, 128, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Oldpeak","opacity":0.8,"orientation":"v","type":"histogram","x":[0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,1.5,0.0,0.0,2.0,0.0,1.0,0.0,1.5,0.0,0.0,1.0,3.0,0.0,1.0,0.0,3.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,1.5,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,4.0,0.0,1.0,0.0,0.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,2.0,2.0,0.0,0.5,0.0,0.0,0.0,1.5,0.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,1.5,0.0,0.0,2.0,1.5,1.0,0.0,0.0,2.0,0.0,2.0,2.5,2.5,3.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,3.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,2.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,1.5,0.0,0.0,0.0,2.0,0.0,2.0,1.0,0.0,0.0,0.0,1.0,1.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,2.0,3.0,0.0,2.0,2.0,0.0,1.0,2.0,1.5,2.0,1.0,1.0,0.0,2.0,0.0,1.0,2.0,0.0,0.0,0.0,0.5,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,2.0,0.0,0.0,3.0,0.0,0.0,0.0,2.0,1.5,0.8,0.0,0.0,2.0,2.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,1.0,0.0,0.0,0.7,1.5,0.7,1.4,0.0,2.1,0.4,0.2,1.5,1.7,2.2,1.5,0.1,0.7,0.5,0.7,1.0,0.1,1.6,0.2,2.0,1.3,0.3,1.8,2.5,1.8,2.6,-0.9,2.8,2.5,-2.6,-1.5,-0.1,0.9,0.8,1.1,2.4,-1.0,-1.1,0.0,-0.7,-0.8,1.6,3.7,2.0,1.1,1.5,1.3,1.4,0.0,0.0,0.0,0.0,0.0,1.6,1.0,0.0,0.5,-1.0,1.0,0.3,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,0.0,2.0,2.0,0.5,2.0,0.0,1.0,0.0,0.0,1.0,1.2,2.0,0.0,0.5,0.5,2.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,1.2,0.0,-0.5,0.0,0.0,2.0,1.5,1.0,-2.0,3.0,0.0,3.0,0.0,1.5,2.5,1.3,-0.5,0.0,1.5,2.0,0.5,0.0,1.0,0.5,1.0,1.0,0.0,2.5,2.0,1.5,0.0,1.0,2.0,0.0,0.2,3.0,1.0,1.2,0.5,1.5,1.6,1.4,2.0,1.0,1.5,2.0,1.0,1.5,2.0,1.2,1.5,0.0,0.0,1.5,0.0,1.9,0.0,1.3,0.0,2.0,0.0,2.5,0.1,1.6,2.0,0.0,3.0,1.5,1.7,0.1,0.0,0.1,2.0,2.0,2.5,2.0,2.5,2.5,1.5,1.1,1.2,0.4,2.0,0.3,3.0,1.0,0.0,3.0,1.7,2.5,1.0,1.0,3.0,0.0,1.0,4.0,2.0,2.0,0.2,3.0,1.2,3.0,0.0,1.5,0.0,0.3,2.0,-0.1,1.3,0.5,3.0,0.0,1.5,1.0,1.0,0.5,4.0,1.0,1.0,0.0,0.1,1.7,0.3,1.5,1.4,1.1,1.8,0.0,2.0,2.5,1.0,1.2,4.0,2.0,0.0,1.2,3.5,1.5,3.0,0.0,0.2,0.0,1.5,1.5,0.2,2.0,0.0,1.8,1.8,0.3,0.0,2.0,1.8,1.4,4.0,0.2,0.1,2.0,1.1,2.0,1.7,1.5,0.0,1.5,2.5,2.0,1.5,0.5,1.5,1.5,1.2,3.0,1.9,3.0,1.8,1.0,1.5,0.0,0.3,1.5,0.8,2.0,1.0,2.0,0.0,0.2,0.0,2.0,0.0,1.0,0.5,0.0,0.2,1.7,1.5,1.0,1.3,0.0,1.5,0.0,1.0,3.0,1.5,0.0,0.0,0.0,0.2,0.0,0.3,0.0,2.4,1.6,0.3,0.2,0.2,0.4,0.6,1.2,1.2,4.0,0.5,0.0,0.0,2.6,0.0,1.6,1.8,3.1,1.8,1.4,2.6,0.2,1.2,0.1,0.0,0.2,0.0,0.6,2.5,0.0,0.4,2.3,0.0,3.4,0.9,0.0,1.9,0.0,0.0,0.0,0.0,0.0,0.4,0.0,2.2,0.0,0.8,0.0,0.0,1.0,1.8,0.0,0.8,0.0,0.6,0.0,3.6,0.0,0.0,1.4,0.2,1.2,0.0,0.9,2.3,0.6,0.0,0.0,0.3,0.0,3.6,0.6,0.0,1.1,0.3,0.0,3.0,0.0,0.0,0.8,2.0,1.6,0.8,2.0,1.5,0.8,0.0,4.2,0.0,2.6,0.0,0.0,2.2,0.0,1.0,1.0,0.4,0.1,0.2,1.1,0.6,1.0,0.0,1.0,1.4,0.5,1.2,2.6,0.0,0.0,3.4,0.0,0.0,0.0,0.0,0.0,0.8,4.0,2.6,1.6,2.0,3.2,1.2,0.8,0.5,0.0,1.8,0.1,0.8,1.4,1.8,0.1,0.0,2.2,1.6,1.4,0.0,1.2,0.7,0.0,2.0,0.0,0.6,1.4,0.0,2.0,0.0,2.0,3.2,0.0,0.0,1.6,0.0,2.0,0.5,0.0,5.6,0.0,1.9,1.0,3.8,1.4,0.0,3.0,0.0,0.0,0.0,1.2,0.2,1.4,0.1,2.0,0.9,1.5,0.0,1.9,4.2,3.6,0.2,0.0,0.8,1.9,0.0,0.6,0.0,1.9,2.1,0.1,1.2,2.9,1.2,2.6,0.0,0.0,0.0,1.4,1.0,1.6,1.8,0.0,1.0,0.0,2.8,1.6,0.8,1.2,0.0,0.6,1.8,3.5,0.2,2.4,0.2,2.2,0.0,1.4,0.0,0.0,0.4,0.0,2.8,2.8,1.6,1.8,1.4,0.0,1.2,3.0,1.0,0.0,1.0,1.2,0.0,0.0,1.8,6.2,0.0,2.5,0.0,0.2,1.6,0.0,0.4,3.6,1.5,1.4,0.6,0.8,3.0,2.8,1.4,0.0,0.0,0.6,1.6,0.4,1.0,1.2,0.0,1.5,0.0,2.4,1.8,0.6,1.0,0.5,0.0,1.3,0.4,1.5,0.0,0.0,0.1,1.0,0.8,0.6,0.0,0.0,0.0,0.6,3.0,0.0,2.0,0.0,0.0,4.4,2.8,0.4,0.0,0.0,0.8,1.2,2.8,4.0,0.0,0.0,1.0,0.2,1.2,3.4,1.2,0.0,0.0]}],                        {"barmode":"overlay","legend":{"bgcolor":"#F5F6F9","font":{"color":"#4D5663"}},"paper_bgcolor":"#F5F6F9","plot_bgcolor":"#F5F6F9","template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"font":{"color":"#4D5663"}},"xaxis":{"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis":{"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"}},                        {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}                    ).then(function(){

var gd = document.getElementById('cf355976-6b4c-42ce-930f-6addee9d3ac9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
df[numerical].iplot(kind='histogram',subplots=True,bins=50)
```


<div>                            <div id="f48dd648-7325-41ff-b058-6c1af3b4f795" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';                                    if (document.getElementById("f48dd648-7325-41ff-b058-6c1af3b4f795")) {                    Plotly.newPlot(                        "f48dd648-7325-41ff-b058-6c1af3b4f795",                        [{"histfunc":"count","histnorm":"","marker":{"color":"rgba(255, 153, 51, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Age","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[40,49,37,48,54,39,45,54,37,48,37,58,39,49,42,54,38,43,60,36,43,44,49,44,40,36,53,52,53,51,53,56,54,41,43,32,65,41,48,48,54,54,35,52,43,59,37,50,36,41,50,47,45,41,52,51,31,58,54,52,49,43,45,46,50,37,45,32,52,44,57,44,52,44,55,46,32,35,52,49,55,54,63,52,56,66,65,53,43,55,49,39,52,48,39,58,43,39,56,41,65,51,40,40,46,57,48,34,50,39,59,57,47,38,49,33,38,59,35,34,47,52,46,58,58,54,34,48,54,42,38,46,56,56,61,49,43,39,54,43,52,50,47,53,56,39,42,43,50,54,39,48,40,55,41,56,38,49,44,54,59,49,47,42,52,46,50,48,58,58,29,40,53,49,52,43,54,59,37,46,52,51,52,46,54,58,58,41,50,53,46,50,48,45,41,62,49,42,53,57,47,46,42,31,56,50,35,35,28,54,48,50,56,56,47,30,39,54,55,29,46,51,48,33,55,50,53,38,41,37,37,40,38,41,54,39,41,55,48,48,55,54,55,43,48,54,54,48,45,49,44,48,61,62,55,53,55,36,51,55,46,54,46,59,47,54,52,34,54,47,45,32,55,55,45,59,51,52,57,54,60,49,51,55,42,51,59,53,48,36,48,47,53,65,32,61,50,57,51,47,60,55,53,62,51,51,55,53,58,57,65,60,41,34,53,74,57,56,61,68,59,63,38,62,46,42,45,59,52,60,60,56,38,40,51,62,72,63,63,64,43,64,61,52,51,69,59,48,69,36,53,43,56,58,55,67,46,53,38,53,62,47,56,56,56,64,61,68,57,63,60,66,63,59,61,73,47,65,70,50,60,50,43,38,54,61,42,53,55,61,51,70,61,38,57,38,62,58,52,61,50,51,65,52,47,35,57,62,59,53,62,54,56,56,54,66,63,44,60,55,66,66,65,60,60,60,56,59,62,63,57,62,63,46,63,60,58,64,63,74,52,69,51,60,56,55,54,77,63,55,52,64,60,60,58,59,61,40,61,41,57,63,59,51,59,42,55,63,62,56,53,68,53,60,62,59,51,61,57,56,58,69,67,58,65,63,55,57,65,54,72,75,49,51,60,64,58,61,67,62,65,63,69,51,62,55,75,40,67,58,60,63,35,62,43,63,68,65,48,63,64,61,50,59,55,45,65,61,49,72,50,64,55,63,59,56,62,74,54,57,62,76,54,70,61,48,48,61,66,68,55,62,71,74,53,58,75,56,58,64,54,54,59,55,57,61,41,71,38,55,56,69,64,72,69,56,62,67,57,69,51,48,69,69,64,57,53,37,67,74,63,58,61,64,58,60,57,55,55,56,57,61,61,74,68,51,62,53,62,46,54,62,55,58,62,70,67,57,64,74,65,56,59,60,63,59,53,44,61,57,71,46,53,64,40,67,48,43,47,54,48,46,51,58,71,57,66,37,59,50,48,61,59,42,48,40,62,44,46,59,58,49,44,66,65,42,52,65,63,45,41,61,60,59,62,57,51,44,60,63,57,51,58,44,47,61,57,70,76,67,45,45,39,42,56,58,35,58,41,57,42,62,59,41,50,59,61,54,54,52,47,66,58,64,50,44,67,49,57,63,48,51,60,59,45,55,41,60,54,42,49,46,56,66,56,49,54,57,65,54,54,62,52,52,60,63,66,42,64,54,46,67,56,34,57,64,59,50,51,54,53,52,40,58,41,41,50,54,64,51,46,55,45,56,66,38,62,55,58,43,64,50,53,45,65,69,69,67,68,34,62,51,46,67,50,42,56,41,42,53,43,56,52,62,70,54,70,54,35,48,55,58,54,69,77,68,58,60,51,55,52,60,58,64,37,59,51,43,58,29,41,63,51,54,44,54,65,57,63,35,41,62,43,58,52,61,39,45,52,62,62,53,43,47,52,68,39,53,62,51,60,65,65,60,60,54,44,44,51,59,71,61,55,64,43,58,60,58,49,48,52,44,56,57,67,53,52,43,52,59,64,66,39,57,58,57,47,55,35,61,58,58,58,56,56,67,55,44,63,63,41,59,57,45,68,57,57,38],"xaxis":"x","yaxis":"y"},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(55, 128, 191, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"RestingBP","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[140,160,130,138,150,120,130,110,140,120,130,136,120,140,115,120,110,120,100,120,100,120,124,150,130,130,124,120,113,125,145,130,125,130,150,125,140,110,120,150,150,130,150,140,120,130,120,140,112,110,130,120,140,130,130,160,120,130,150,112,100,150,140,120,110,120,132,110,160,150,140,130,120,120,140,150,118,140,140,130,110,120,150,160,150,140,170,140,120,140,110,130,120,160,110,130,142,160,120,125,130,130,150,120,118,140,120,150,140,190,130,150,140,140,130,100,120,130,120,140,135,125,110,180,130,120,130,108,120,120,145,110,170,150,130,115,120,120,140,150,160,140,160,140,120,110,120,120,120,130,130,100,130,120,120,155,110,140,130,160,140,128,160,120,140,140,140,140,135,140,120,140,140,140,140,140,140,140,130,130,130,130,140,110,160,160,130,120,120,180,180,170,130,135,125,160,120,150,120,130,110,120,160,100,130,150,120,110,130,125,106,140,130,130,150,170,110,120,140,140,130,160,120,120,120,145,120,92,120,130,130,130,120,112,140,120,120,140,160,160,145,200,160,120,160,120,120,122,130,130,135,120,125,140,145,120,130,150,150,122,140,120,120,130,140,160,130,98,130,130,120,105,140,120,180,180,135,170,180,130,120,150,130,110,140,110,140,120,133,120,110,140,130,115,95,105,145,110,110,110,160,140,125,120,95,120,115,130,115,95,155,125,125,115,80,145,105,140,130,145,125,100,105,115,100,105,110,125,95,130,115,115,100,95,130,120,160,150,140,95,100,110,110,130,120,135,120,115,137,110,120,140,120,130,120,145,115,120,115,105,160,160,155,120,120,200,150,135,140,150,135,150,185,135,125,160,155,160,140,120,160,115,115,110,120,150,145,130,140,160,140,115,130,150,160,135,140,170,165,200,160,130,145,135,110,120,140,115,110,160,150,180,125,125,130,155,140,130,132,142,110,120,150,180,120,160,126,140,110,133,128,120,170,110,126,152,116,120,130,138,128,130,128,130,120,136,130,124,160,0,122,144,140,120,136,154,120,125,134,104,139,136,122,128,131,134,120,132,152,124,126,138,154,141,131,178,132,110,130,170,126,140,142,120,134,139,110,140,140,136,120,170,130,137,142,142,132,146,160,135,136,130,140,132,158,136,136,106,120,110,136,160,123,112,122,130,150,150,102,96,130,120,144,124,150,130,144,139,131,143,133,143,116,110,125,130,133,150,130,110,138,104,138,170,140,132,132,142,112,139,172,120,144,145,155,150,160,137,137,134,133,132,140,135,144,141,150,130,110,158,128,140,150,160,142,137,139,146,156,145,131,140,122,142,141,180,124,118,140,140,136,100,190,130,160,130,122,133,120,130,130,140,120,155,134,114,160,144,158,134,127,135,122,140,120,130,115,124,128,120,120,130,110,140,150,135,142,140,134,128,112,140,140,110,140,120,130,115,112,132,130,138,120,112,110,128,160,120,170,144,130,140,160,130,122,152,124,130,101,126,140,118,110,160,150,136,128,140,140,130,105,138,120,174,120,150,130,120,150,145,150,140,136,118,108,120,120,156,140,106,142,104,94,120,120,146,120,150,130,110,148,128,178,126,150,140,130,124,110,125,110,120,100,140,120,108,120,130,165,130,124,100,150,140,112,180,110,158,135,120,134,120,200,150,130,120,122,152,160,125,160,120,136,134,117,108,112,140,120,150,142,152,125,118,132,145,138,140,125,192,123,112,110,132,112,112,120,108,130,130,105,140,128,120,178,120,150,130,128,110,180,110,130,138,138,160,140,100,120,118,138,140,150,125,129,120,134,110,102,130,130,132,108,140,160,140,145,108,126,124,135,100,110,140,125,118,125,125,140,160,152,102,105,125,130,170,125,122,128,130,130,135,94,120,120,110,135,150,130,138,135,130,132,150,118,145,118,115,128,130,160,138,120,138,120,180,140,130,140,140,130,110,155,140,145,120,130,112,110,150,160,150,132,140,150,120,130,120,130,110,172,120,140,140,160,128,138,132,128,134,170,146,138,154,130,110,130,128,122,148,114,170,125,130,120,152,132,120,140,124,120,164,140,110,144,130,130,138],"xaxis":"x2","yaxis":"y2"},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(50, 171, 96, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Cholesterol","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[289,180,283,214,195,339,237,208,207,284,211,164,204,234,211,273,196,201,248,267,223,184,201,288,215,209,260,284,468,188,518,167,224,172,186,254,306,250,177,227,230,294,264,259,175,318,223,216,340,289,233,205,224,245,180,194,270,213,365,342,253,254,224,277,202,260,297,225,246,412,265,215,182,218,268,163,529,167,100,206,277,238,223,196,213,139,263,216,291,229,208,307,210,329,182,263,207,147,85,269,275,179,392,466,186,260,254,214,129,241,188,255,276,297,207,246,282,338,160,156,248,272,240,393,230,246,161,163,230,228,292,202,388,230,294,265,215,241,166,247,331,341,291,243,279,273,198,249,168,603,215,159,275,270,291,342,190,185,290,195,264,212,263,196,225,272,231,238,222,179,243,235,320,187,266,288,216,287,194,238,225,224,404,238,312,211,251,237,328,285,280,209,245,192,184,193,297,268,246,308,249,230,147,219,184,215,308,257,132,216,263,288,276,219,226,237,280,217,196,263,222,303,195,298,256,264,195,117,295,173,315,281,275,250,309,200,336,295,355,193,326,198,292,266,268,171,237,275,219,341,491,260,292,271,248,274,394,160,200,320,275,221,231,126,193,305,298,220,242,235,225,198,201,220,295,213,160,223,347,253,246,222,220,344,358,190,169,181,308,166,211,257,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,260,209,218,228,213,0,236,0,0,267,166,0,0,0,0,220,177,236,0,0,0,0,0,0,0,0,0,186,100,228,0,171,230,0,0,0,281,0,203,0,0,0,0,0,277,0,233,0,0,240,0,0,153,224,0,0,0,316,0,0,218,0,311,0,0,0,270,0,0,217,214,214,252,220,214,203,0,339,216,276,458,241,384,297,248,308,208,227,210,245,225,240,0,198,195,267,161,258,0,0,195,235,0,305,223,282,349,160,160,236,312,283,142,211,218,306,186,252,222,0,0,258,202,197,204,113,274,192,298,272,220,200,261,181,260,220,221,216,175,219,310,208,232,273,203,182,274,204,270,292,171,221,289,217,223,110,193,123,210,282,170,369,173,289,152,208,216,271,244,285,243,240,219,237,165,213,287,258,256,186,264,185,226,203,207,284,337,310,254,258,254,300,170,310,333,139,223,385,254,322,564,261,263,269,177,256,239,293,407,234,226,235,234,303,149,311,203,211,199,229,245,303,204,288,275,243,295,230,265,229,228,215,326,200,256,207,273,180,222,223,209,233,197,218,211,149,197,246,225,315,205,417,195,234,198,166,178,249,281,126,305,226,240,233,276,261,319,242,243,260,354,245,197,223,309,208,199,209,236,218,198,270,214,201,244,208,270,306,243,221,330,266,206,212,275,302,234,313,244,141,237,269,289,254,274,222,258,177,160,327,235,305,304,295,271,249,288,226,283,188,286,274,360,273,201,267,196,201,230,269,212,226,246,232,177,277,249,210,207,212,271,233,213,283,282,230,167,224,268,250,219,267,303,256,204,217,308,193,228,231,244,262,259,211,325,254,197,236,282,234,254,299,211,182,294,298,231,254,196,240,409,172,265,246,315,184,233,394,269,239,174,309,282,255,250,248,214,239,304,277,300,258,299,289,298,318,240,309,250,288,245,213,216,204,204,252,227,258,220,239,254,168,330,183,203,263,341,283,186,307,219,260,255,231,164,234,177,257,325,274,321,264,268,308,253,248,269,185,282,188,219,290,175,212,302,243,353,335,247,340,206,284,266,229,199,263,294,192,286,216,223,247,204,204,227,278,220,232,197,335,253,205,192,203,318,225,220,221,240,212,342,169,187,197,157,176,241,264,193,131,236,175],"xaxis":"x3","yaxis":"y3"},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(128, 0, 128, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"FastingBS","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],"xaxis":"x4","yaxis":"y4"},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(219, 64, 82, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"MaxHR","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[172,156,98,108,122,170,170,142,130,120,142,99,145,140,137,150,166,165,125,160,142,142,164,150,138,178,112,118,127,145,130,114,122,130,154,155,87,142,148,130,130,100,168,170,120,120,168,170,184,170,121,98,122,150,140,170,153,140,134,96,174,175,144,125,145,130,144,184,82,170,145,135,150,115,128,116,130,150,138,170,160,154,115,165,125,94,112,142,155,110,160,140,148,92,180,140,138,160,140,144,115,100,130,152,124,140,110,168,135,106,124,92,125,150,135,150,170,130,185,180,170,139,140,110,150,110,190,175,140,152,130,150,122,124,120,175,175,146,118,130,94,125,158,155,150,132,155,176,160,125,120,100,150,140,160,150,150,130,100,130,119,96,174,150,140,175,140,118,100,160,160,188,162,172,134,135,105,150,150,90,120,150,124,140,130,92,110,138,110,120,120,116,160,110,180,116,132,136,116,98,150,150,146,150,100,140,180,140,185,140,110,140,128,164,98,170,150,137,150,170,112,150,125,185,137,150,140,134,170,184,158,167,129,142,140,160,118,136,99,102,155,142,143,118,103,137,150,150,130,120,135,115,115,152,96,130,150,172,120,155,165,138,115,125,145,175,110,150,91,145,140,165,130,134,180,100,150,126,126,155,135,122,160,160,170,120,140,132,156,180,138,135,148,93,127,110,139,131,92,149,149,150,120,123,126,127,155,120,138,182,154,110,176,154,141,123,148,121,77,136,175,109,166,128,133,128,138,119,82,130,143,82,179,144,170,134,114,154,149,145,122,114,113,120,104,130,115,128,104,125,120,140,100,100,92,125,113,95,128,115,72,124,99,148,97,140,117,120,120,86,63,108,98,115,105,121,118,122,157,156,99,120,145,156,155,105,99,135,83,145,60,92,115,120,98,150,143,105,122,70,110,163,67,128,120,130,100,72,94,122,78,150,103,98,110,90,112,127,140,149,99,120,105,140,141,157,140,117,120,120,148,86,84,125,120,118,124,106,111,116,180,129,125,140,120,124,117,110,105,155,110,122,118,133,123,131,80,165,86,111,118,84,117,107,128,160,125,130,97,161,106,130,140,122,130,120,139,108,148,123,110,118,125,106,112,128,180,144,135,140,102,108,145,127,110,140,69,148,130,130,140,138,140,138,112,131,112,80,150,110,126,88,153,150,120,160,132,120,110,121,128,135,120,117,150,144,113,135,127,109,128,115,102,140,135,122,119,130,112,100,122,120,105,129,120,139,162,100,140,135,73,86,108,116,160,118,112,122,124,102,137,141,154,126,160,115,128,115,105,110,119,109,135,130,112,126,120,110,119,110,130,159,84,126,116,120,122,165,122,94,133,110,150,130,113,140,100,136,127,98,96,123,98,112,151,96,108,128,138,126,154,137,100,135,93,109,160,141,105,121,140,142,142,170,154,161,111,180,145,159,125,120,155,144,178,129,180,181,143,159,139,152,157,165,130,150,138,170,140,126,150,138,125,150,186,181,163,179,156,134,165,126,177,120,114,125,184,157,179,175,168,125,96,143,103,173,142,169,171,150,112,186,152,149,152,140,163,143,116,142,147,148,179,173,178,105,130,111,168,126,178,140,145,163,128,164,169,109,108,168,118,151,156,133,162,175,71,163,124,147,166,143,157,162,138,117,153,161,170,162,162,144,133,114,103,139,116,88,151,152,163,99,169,158,160,169,132,178,96,165,160,172,144,192,168,132,182,163,125,195,95,160,114,173,172,179,158,167,122,149,172,111,170,162,165,182,154,155,130,161,154,159,152,152,174,131,146,125,115,174,106,122,147,163,163,194,150,158,122,173,162,105,147,157,112,160,125,156,156,175,161,122,158,151,162,151,171,141,173,145,178,160,154,131,187,159,166,165,131,202,172,172,154,147,170,126,127,174,132,182,132,97,136,162,190,146,140,185,161,146,145,160,120,156,172,150,182,143,160,142,144,158,148,155,142,113,188,153,123,157,162,137,132,158,171,172,132,160,171,168,162,173,153,148,108,115,169,143,156,162,155,152,152,164,131,143,179,130,174,161,140,146,144,163,169,150,166,144,144,136,182,90,123,132,141,115,174,173],"xaxis":"x5","yaxis":"y5"},{"histfunc":"count","histnorm":"","marker":{"color":"rgba(0, 128, 128, 1.0)","line":{"color":"#4D5663","width":1.3}},"name":"Oldpeak","nbinsx":50,"opacity":0.8,"orientation":"v","type":"histogram","x":[0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,1.5,0.0,0.0,2.0,0.0,1.0,0.0,1.5,0.0,0.0,1.0,3.0,0.0,1.0,0.0,3.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,2.0,2.0,0.0,0.0,1.5,0.0,1.5,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,4.0,0.0,1.0,0.0,0.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,2.0,2.0,0.0,0.5,0.0,0.0,0.0,1.5,0.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,1.5,0.0,0.0,2.0,1.5,1.0,0.0,0.0,2.0,0.0,2.0,2.5,2.5,3.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,3.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,2.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,1.5,0.0,0.0,0.0,2.0,0.0,2.0,1.0,0.0,0.0,0.0,1.0,1.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,0.0,2.0,3.0,0.0,2.0,2.0,0.0,1.0,2.0,1.5,2.0,1.0,1.0,0.0,2.0,0.0,1.0,2.0,0.0,0.0,0.0,0.5,0.0,0.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,2.0,0.0,0.0,3.0,0.0,0.0,0.0,2.0,1.5,0.8,0.0,0.0,2.0,2.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,1.0,0.0,0.0,0.7,1.5,0.7,1.4,0.0,2.1,0.4,0.2,1.5,1.7,2.2,1.5,0.1,0.7,0.5,0.7,1.0,0.1,1.6,0.2,2.0,1.3,0.3,1.8,2.5,1.8,2.6,-0.9,2.8,2.5,-2.6,-1.5,-0.1,0.9,0.8,1.1,2.4,-1.0,-1.1,0.0,-0.7,-0.8,1.6,3.7,2.0,1.1,1.5,1.3,1.4,0.0,0.0,0.0,0.0,0.0,1.6,1.0,0.0,0.5,-1.0,1.0,0.3,0.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,0.0,2.0,2.0,0.5,2.0,0.0,1.0,0.0,0.0,1.0,1.2,2.0,0.0,0.5,0.5,2.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.7,2.0,0.0,1.2,0.0,-0.5,0.0,0.0,2.0,1.5,1.0,-2.0,3.0,0.0,3.0,0.0,1.5,2.5,1.3,-0.5,0.0,1.5,2.0,0.5,0.0,1.0,0.5,1.0,1.0,0.0,2.5,2.0,1.5,0.0,1.0,2.0,0.0,0.2,3.0,1.0,1.2,0.5,1.5,1.6,1.4,2.0,1.0,1.5,2.0,1.0,1.5,2.0,1.2,1.5,0.0,0.0,1.5,0.0,1.9,0.0,1.3,0.0,2.0,0.0,2.5,0.1,1.6,2.0,0.0,3.0,1.5,1.7,0.1,0.0,0.1,2.0,2.0,2.5,2.0,2.5,2.5,1.5,1.1,1.2,0.4,2.0,0.3,3.0,1.0,0.0,3.0,1.7,2.5,1.0,1.0,3.0,0.0,1.0,4.0,2.0,2.0,0.2,3.0,1.2,3.0,0.0,1.5,0.0,0.3,2.0,-0.1,1.3,0.5,3.0,0.0,1.5,1.0,1.0,0.5,4.0,1.0,1.0,0.0,0.1,1.7,0.3,1.5,1.4,1.1,1.8,0.0,2.0,2.5,1.0,1.2,4.0,2.0,0.0,1.2,3.5,1.5,3.0,0.0,0.2,0.0,1.5,1.5,0.2,2.0,0.0,1.8,1.8,0.3,0.0,2.0,1.8,1.4,4.0,0.2,0.1,2.0,1.1,2.0,1.7,1.5,0.0,1.5,2.5,2.0,1.5,0.5,1.5,1.5,1.2,3.0,1.9,3.0,1.8,1.0,1.5,0.0,0.3,1.5,0.8,2.0,1.0,2.0,0.0,0.2,0.0,2.0,0.0,1.0,0.5,0.0,0.2,1.7,1.5,1.0,1.3,0.0,1.5,0.0,1.0,3.0,1.5,0.0,0.0,0.0,0.2,0.0,0.3,0.0,2.4,1.6,0.3,0.2,0.2,0.4,0.6,1.2,1.2,4.0,0.5,0.0,0.0,2.6,0.0,1.6,1.8,3.1,1.8,1.4,2.6,0.2,1.2,0.1,0.0,0.2,0.0,0.6,2.5,0.0,0.4,2.3,0.0,3.4,0.9,0.0,1.9,0.0,0.0,0.0,0.0,0.0,0.4,0.0,2.2,0.0,0.8,0.0,0.0,1.0,1.8,0.0,0.8,0.0,0.6,0.0,3.6,0.0,0.0,1.4,0.2,1.2,0.0,0.9,2.3,0.6,0.0,0.0,0.3,0.0,3.6,0.6,0.0,1.1,0.3,0.0,3.0,0.0,0.0,0.8,2.0,1.6,0.8,2.0,1.5,0.8,0.0,4.2,0.0,2.6,0.0,0.0,2.2,0.0,1.0,1.0,0.4,0.1,0.2,1.1,0.6,1.0,0.0,1.0,1.4,0.5,1.2,2.6,0.0,0.0,3.4,0.0,0.0,0.0,0.0,0.0,0.8,4.0,2.6,1.6,2.0,3.2,1.2,0.8,0.5,0.0,1.8,0.1,0.8,1.4,1.8,0.1,0.0,2.2,1.6,1.4,0.0,1.2,0.7,0.0,2.0,0.0,0.6,1.4,0.0,2.0,0.0,2.0,3.2,0.0,0.0,1.6,0.0,2.0,0.5,0.0,5.6,0.0,1.9,1.0,3.8,1.4,0.0,3.0,0.0,0.0,0.0,1.2,0.2,1.4,0.1,2.0,0.9,1.5,0.0,1.9,4.2,3.6,0.2,0.0,0.8,1.9,0.0,0.6,0.0,1.9,2.1,0.1,1.2,2.9,1.2,2.6,0.0,0.0,0.0,1.4,1.0,1.6,1.8,0.0,1.0,0.0,2.8,1.6,0.8,1.2,0.0,0.6,1.8,3.5,0.2,2.4,0.2,2.2,0.0,1.4,0.0,0.0,0.4,0.0,2.8,2.8,1.6,1.8,1.4,0.0,1.2,3.0,1.0,0.0,1.0,1.2,0.0,0.0,1.8,6.2,0.0,2.5,0.0,0.2,1.6,0.0,0.4,3.6,1.5,1.4,0.6,0.8,3.0,2.8,1.4,0.0,0.0,0.6,1.6,0.4,1.0,1.2,0.0,1.5,0.0,2.4,1.8,0.6,1.0,0.5,0.0,1.3,0.4,1.5,0.0,0.0,0.1,1.0,0.8,0.6,0.0,0.0,0.0,0.6,3.0,0.0,2.0,0.0,0.0,4.4,2.8,0.4,0.0,0.0,0.8,1.2,2.8,4.0,0.0,0.0,1.0,0.2,1.2,3.4,1.2,0.0,0.0],"xaxis":"x6","yaxis":"y6"}],                        {"barmode":"overlay","legend":{"bgcolor":"#F5F6F9","font":{"color":"#4D5663"}},"paper_bgcolor":"#F5F6F9","plot_bgcolor":"#F5F6F9","template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"font":{"color":"#4D5663"}},"xaxis":{"anchor":"y","domain":[0.0,0.45],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"xaxis2":{"anchor":"y2","domain":[0.55,1.0],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"xaxis3":{"anchor":"y3","domain":[0.0,0.45],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"xaxis4":{"anchor":"y4","domain":[0.55,1.0],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"xaxis5":{"anchor":"y5","domain":[0.0,0.45],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"xaxis6":{"anchor":"y6","domain":[0.55,1.0],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis":{"anchor":"x","domain":[0.7333333333333333,1.0],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis2":{"anchor":"x2","domain":[0.7333333333333333,1.0],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis3":{"anchor":"x3","domain":[0.36666666666666664,0.6333333333333333],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis4":{"anchor":"x4","domain":[0.36666666666666664,0.6333333333333333],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis5":{"anchor":"x5","domain":[0.0,0.26666666666666666],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"},"yaxis6":{"anchor":"x6","domain":[0.0,0.26666666666666666],"gridcolor":"#E1E5ED","showgrid":true,"tickfont":{"color":"#4D5663"},"title":{"font":{"color":"#4D5663"},"text":""},"zerolinecolor":"#E1E5ED"}},                        {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}                    ).then(function(){

var gd = document.getElementById('f48dd648-7325-41ff-b058-6c1af3b4f795');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
skew_limit = 0.75 # This is our threshold-limit to evaluate skewness. Overall below abs(1) seems acceptable for the linear models. 
skew_vals = df[numerical].drop('FastingBS', axis=1).skew()
skew_cols= skew_vals[abs(skew_vals)> skew_limit].sort_values(ascending=False)
skew_cols
```




    Oldpeak    1.022872
    dtype: float64



- Nothing much for the skewness. Quite a normal like distribution for the numerical features.


```python
numerical1= df.select_dtypes('number').columns


matrix = np.triu(df[numerical1].corr())
fig, ax = plt.subplots(figsize=(14,10)) 
sns.heatmap (df[numerical1].corr(), annot=True, fmt= '.2f', vmin=-1, vmax=1, center=0, cmap='coolwarm',mask=matrix, ax=ax);
```


    
![png](notebook_files/notebook_31_0.png)
    


- Based on the  matrix, we can observe weak level correlation between the numerical features and the target variable
- Oldpeak (depression related number) has a positive correlation with the heart disease.
- Maximum heart rate has negative correlation with the heart disease.
- interestingly cholesterol has negative correlation with the heart disease.


<a id="5"></a>
<font color="lightseagreen" size=+1.5><b>Categorical Features</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
df[categorical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingECG</th>
      <th>ExerciseAngina</th>
      <th>ST_Slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>ATA</td>
      <td>Normal</td>
      <td>N</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>NAP</td>
      <td>Normal</td>
      <td>N</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>ATA</td>
      <td>ST</td>
      <td>N</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>ASY</td>
      <td>Normal</td>
      <td>Y</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>NAP</td>
      <td>Normal</td>
      <td>N</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
</div>



### **Gender and Heart Disease**


```python
print (f'A female person has a probability of {round(df[df["Sex"]=="F"]["HeartDisease"].mean()*100,2)} % have a HeartDisease')

print()

print (f'A male person has a probability of {round(df[df["Sex"]=="M"]["HeartDisease"].mean()*100,2)} % have a HeartDisease')

print()

```

    A female person has a probability of 25.91 % have a HeartDisease
    
    A male person has a probability of 63.17 % have a HeartDisease
    
    


```python
fig = px.histogram(df, x="Sex", color="HeartDisease",width=400, height=400)
fig.show()
```


<div>                            <div id="6de895ae-ca85-49e6-a0ff-a07871d5e794" class="plotly-graph-div" style="height:400px; width:400px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6de895ae-ca85-49e6-a0ff-a07871d5e794")) {                    Plotly.newPlot(                        "6de895ae-ca85-49e6-a0ff-a07871d5e794",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=0<br>Sex=%{x}<br>count=%{y}<extra></extra>","legendgroup":"0","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"0","offsetgroup":"0","orientation":"v","showlegend":true,"type":"histogram","x":["M","M","M","M","F","M","F","F","M","F","F","F","F","M","F","M","M","M","M","F","M","M","F","M","F","F","F","F","M","M","M","M","M","M","M","F","F","F","M","M","F","M","F","F","F","M","M","M","F","M","F","M","M","M","M","F","M","F","M","F","F","M","M","M","M","M","M","F","M","M","M","F","M","M","F","F","F","M","M","M","F","F","F","M","M","F","F","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","F","M","M","F","M","F","M","M","M","M","F","F","M","M","F","M","M","M","F","M","M","M","M","M","F","F","M","M","M","F","M","M","M","M","F","M","M","F","M","F","M","M","F","M","M","M","F","F","M","F","F","M","F","M","M","F","M","F","M","F","M","F","F","M","F","F","M","F","M","M","F","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","F","M","F","M","M","M","M","F","F","M","M","M","M","M","F","F","F","F","F","M","F","M","M","F","M","M","M","M","F","F","F","F","F","M","M","F","M","M","F","F","M","F","F","M","F","F","M","M","M","M","M","M","F","M","M","F","F","F","F","M","F","M","F","M","F","F","F","M","F","F","M","M","M","F","F","F","M","M","M","F","M","F","F","F","F","F","M","M","M","M","F","F","M","F","M","F","M","M","M","F","M","M","M","F","M","M","M","F","F","F","F","M","M","F","M","M","M","F","M","F","F","M","M","M","M","F","M","F","M","M","M","F","M","M","M","F","F","F","M","M","M","F","M","M","F","M","M","M","F","M","F","M","M","F","F","F","M","M","M","M","M","F","M","M"],"xaxis":"x","yaxis":"y"},{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=1<br>Sex=%{x}<br>count=%{y}<extra></extra>","legendgroup":"1","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"1","offsetgroup":"1","orientation":"v","showlegend":true,"type":"histogram","x":["F","F","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","F","M","F","M","F","F","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","F","M","F","M","F","M","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","F","M","F","F","M","F","M","M","M","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","F","M","F","M","M","M","M","F","M","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","F","F","M","M","F","M","M","F","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","F","M","M","F","M","M","M","F","M","F","M","M","M","F"],"xaxis":"x","yaxis":"y"}],                        {"barmode":"relative","height":400,"legend":{"title":{"text":"HeartDisease"},"tracegroupgap":0},"margin":{"t":60},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"width":400,"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Sex"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('6de895ae-ca85-49e6-a0ff-a07871d5e794');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- Bad news guys....
- Men are almost 2.44 times more likely have a heart disease than women.


### **Chest Pain Type and Heart Disease**


```python
df.groupby('ChestPainType')['HeartDisease'].mean().sort_values(ascending=False)
```




    ChestPainType
    ASY    0.790323
    TA     0.434783
    NAP    0.354680
    ATA    0.138728
    Name: HeartDisease, dtype: float64




```python
fig = px.histogram(df, x="ChestPainType", color="HeartDisease",width=400, height=400)
fig.show()
```


<div>                            <div id="cb8e6b68-c477-4e2b-959b-92a6d1a65041" class="plotly-graph-div" style="height:400px; width:400px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("cb8e6b68-c477-4e2b-959b-92a6d1a65041")) {                    Plotly.newPlot(                        "cb8e6b68-c477-4e2b-959b-92a6d1a65041",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=0<br>ChestPainType=%{x}<br>count=%{y}<extra></extra>","legendgroup":"0","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"0","offsetgroup":"0","orientation":"v","showlegend":true,"type":"histogram","x":["ATA","ATA","NAP","NAP","ATA","ATA","ATA","NAP","ATA","NAP","ATA","ATA","TA","ATA","ATA","NAP","NAP","ASY","ATA","ATA","ATA","NAP","ATA","ATA","ATA","ATA","ASY","ATA","ATA","NAP","NAP","ASY","ATA","NAP","ATA","ATA","ASY","ATA","ASY","ATA","NAP","ASY","ATA","ATA","ASY","ATA","ASY","ATA","ASY","NAP","ASY","ATA","NAP","ATA","ATA","ATA","ASY","ATA","ASY","ATA","NAP","ATA","NAP","ASY","ATA","ASY","ATA","ASY","ATA","ASY","ATA","ATA","ASY","ATA","NAP","TA","NAP","ASY","ATA","ATA","ATA","ASY","ATA","NAP","NAP","ASY","ATA","ATA","ASY","ASY","ATA","ATA","ATA","ATA","ATA","ATA","ASY","ATA","ASY","ATA","ATA","ATA","ATA","ATA","ASY","NAP","ATA","NAP","ATA","NAP","ATA","NAP","ASY","ATA","ASY","ATA","ATA","ASY","ASY","ATA","ATA","NAP","ATA","TA","ASY","ATA","TA","TA","NAP","NAP","ATA","ATA","ASY","ATA","ATA","NAP","NAP","TA","NAP","ATA","ATA","NAP","NAP","ATA","NAP","ATA","ASY","ASY","NAP","ATA","ASY","ATA","ATA","ATA","ATA","TA","ASY","ATA","NAP","ATA","NAP","NAP","ATA","ATA","ATA","ATA","ATA","NAP","ASY","ATA","NAP","ATA","NAP","ASY","ATA","NAP","NAP","ATA","ASY","NAP","ASY","ATA","ATA","ATA","NAP","ATA","ASY","ATA","ATA","ASY","ASY","NAP","NAP","NAP","NAP","ASY","NAP","NAP","NAP","ATA","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","NAP","ATA","ATA","ATA","ASY","ATA","TA","ASY","ASY","NAP","ASY","NAP","NAP","ASY","ASY","NAP","ATA","ASY","ATA","ASY","TA","NAP","NAP","ASY","NAP","ATA","ASY","ASY","NAP","NAP","ATA","ASY","NAP","ASY","NAP","ASY","TA","ASY","NAP","ASY","ATA","ASY","ASY","ASY","NAP","ASY","ASY","TA","TA","ATA","ASY","ASY","ATA","NAP","ASY","NAP","NAP","ASY","NAP","NAP","ASY","ASY","NAP","ATA","NAP","ATA","NAP","ATA","ATA","ATA","NAP","NAP","NAP","TA","TA","ASY","NAP","ASY","ATA","NAP","ASY","ASY","NAP","NAP","ATA","NAP","ASY","TA","ATA","TA","ATA","ATA","ASY","NAP","ATA","NAP","ASY","NAP","ATA","ATA","NAP","ATA","ATA","TA","NAP","NAP","NAP","NAP","ATA","ASY","NAP","NAP","NAP","ATA","ASY","ASY","TA","NAP","NAP","NAP","NAP","ASY","NAP","ATA","ATA","TA","ATA","ASY","ASY","ASY","NAP","ASY","TA","NAP","TA","NAP","NAP","ASY","NAP","NAP","ASY","ASY","ASY","ATA","NAP","ATA","ASY","NAP","TA","NAP","TA","NAP","NAP","NAP","NAP","NAP","ATA","ATA","NAP","NAP","NAP","ATA","NAP","ASY","ATA","TA","TA","ASY","NAP","ASY","NAP","ATA","NAP","ASY","NAP","NAP","ATA","NAP","NAP","ATA","NAP","ASY","NAP","ATA","NAP","ATA","ATA","ASY","NAP","NAP","TA","NAP","NAP","ASY","NAP","ATA","ATA","ATA","ATA","ATA","ATA","NAP"],"xaxis":"x","yaxis":"y"},{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=1<br>ChestPainType=%{x}<br>count=%{y}<extra></extra>","legendgroup":"1","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"1","offsetgroup":"1","orientation":"v","showlegend":true,"type":"histogram","x":["NAP","ASY","ASY","ATA","ASY","ASY","ASY","ATA","ATA","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ATA","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","TA","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","TA","NAP","ATA","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ATA","ASY","ASY","ASY","ASY","ASY","ASY","TA","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ATA","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","TA","ASY","ASY","ASY","ATA","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","TA","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ATA","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","TA","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","TA","ASY","ATA","NAP","NAP","NAP","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","TA","ASY","ASY","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ATA","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","NAP","ATA","NAP","TA","ASY","ASY","ATA","ASY","ASY","NAP","ASY","TA","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","TA","ASY","NAP","ASY","NAP","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","NAP","NAP","ASY","ASY","NAP","TA","ASY","ASY","NAP","TA","NAP","NAP","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ATA","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ATA","ASY","ATA","NAP","ASY","ASY","ASY","TA","ASY","ASY","ASY","NAP","NAP","ASY","ASY","ASY","ASY","TA","ASY","NAP","NAP","ASY","ATA","ASY","ASY","ASY","ASY","ATA","ASY","ASY","ATA","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ATA","ASY","ASY","ASY","NAP","ASY","ASY","TA","ASY","ASY","TA","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","ASY","NAP","TA","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","NAP","ASY","ATA","ATA","ASY","ASY","ASY","TA","ATA","ASY","ASY","ASY","ASY","ASY","NAP","ASY","ASY","ASY","ASY","ASY","TA","ASY","ASY","ATA"],"xaxis":"x","yaxis":"y"}],                        {"barmode":"relative","height":400,"legend":{"title":{"text":"HeartDisease"},"tracegroupgap":0},"margin":{"t":60},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"width":400,"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"ChestPainType"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('cb8e6b68-c477-4e2b-959b-92a6d1a65041');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- We can observe clear differences among the chest pain type.
- Person with ASY: Asymptomatic chest pain  has almost 6 times more likely have a heart disease than person with ATA Atypical Angina chest pain.


### **RestingECG and Heart Disease**


```python
df.groupby('RestingECG')['HeartDisease'].mean().sort_values(ascending=False)
```




    RestingECG
    ST        0.657303
    LVH       0.563830
    Normal    0.516304
    Name: HeartDisease, dtype: float64




```python
fig = px.histogram(df, x="RestingECG", color="HeartDisease",width=400, height=400)
fig.show()
```


<div>                            <div id="8c32b8a4-c7c6-4781-a765-4c05dae3d97a" class="plotly-graph-div" style="height:400px; width:400px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("8c32b8a4-c7c6-4781-a765-4c05dae3d97a")) {                    Plotly.newPlot(                        "8c32b8a4-c7c6-4781-a765-4c05dae3d97a",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=0<br>RestingECG=%{x}<br>count=%{y}<extra></extra>","legendgroup":"0","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"0","offsetgroup":"0","orientation":"v","showlegend":true,"type":"histogram","x":["Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","ST","ST","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","LVH","LVH","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","ST","Normal","Normal","LVH","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","LVH","Normal","Normal","ST","ST","LVH","Normal","ST","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","LVH","Normal","ST","LVH","ST","ST","ST","ST","ST","ST","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","LVH","ST","Normal","Normal","ST","Normal","Normal","ST","ST","ST","ST","Normal","Normal","LVH","Normal","ST","ST","LVH","ST","Normal","ST","ST","Normal","Normal","LVH","LVH","ST","ST","ST","ST","Normal","LVH","ST","LVH","LVH","Normal","LVH","Normal","Normal","LVH","LVH","LVH","Normal","LVH","Normal","LVH","Normal","Normal","LVH","Normal","LVH","LVH","LVH","LVH","Normal","Normal","LVH","Normal","Normal","Normal","LVH","Normal","LVH","Normal","LVH","Normal","Normal","Normal","Normal","Normal","LVH","LVH","Normal","Normal","LVH","ST","Normal","LVH","Normal","Normal","Normal","LVH","Normal","LVH","LVH","LVH","Normal","Normal","LVH","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","LVH","LVH","Normal","LVH","Normal","Normal","LVH","LVH","Normal","Normal","Normal","LVH","LVH","LVH","Normal","Normal","LVH","Normal","LVH","Normal","LVH","LVH","Normal","Normal","Normal","LVH","LVH","LVH","LVH","LVH","LVH","Normal","Normal","LVH","LVH","Normal","Normal","LVH","Normal","Normal","Normal","LVH","LVH","Normal","Normal","Normal","Normal","Normal","LVH","Normal","LVH","Normal","LVH","LVH","LVH","Normal","LVH","Normal","Normal","Normal","Normal","LVH","LVH","LVH","Normal","LVH","LVH","Normal","LVH","LVH","LVH","Normal","LVH","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","LVH","Normal","LVH","Normal","LVH","LVH","Normal","Normal","Normal","Normal","Normal","LVH","Normal","Normal","Normal","Normal"],"xaxis":"x","yaxis":"y"},{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=1<br>RestingECG=%{x}<br>count=%{y}<extra></extra>","legendgroup":"1","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"1","offsetgroup":"1","orientation":"v","showlegend":true,"type":"histogram","x":["Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","LVH","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","ST","Normal","ST","ST","Normal","LVH","ST","LVH","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","Normal","ST","ST","ST","ST","LVH","Normal","Normal","Normal","Normal","Normal","Normal","ST","ST","Normal","Normal","ST","Normal","Normal","Normal","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","LVH","Normal","ST","Normal","ST","LVH","Normal","Normal","Normal","Normal","ST","Normal","ST","Normal","ST","ST","Normal","Normal","Normal","Normal","Normal","Normal","LVH","ST","Normal","Normal","Normal","Normal","Normal","ST","ST","ST","ST","Normal","ST","ST","Normal","LVH","ST","Normal","Normal","Normal","ST","ST","ST","ST","ST","ST","Normal","ST","ST","ST","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","ST","ST","Normal","Normal","ST","ST","LVH","Normal","ST","LVH","Normal","ST","Normal","LVH","Normal","ST","ST","Normal","Normal","ST","ST","Normal","Normal","Normal","Normal","ST","Normal","Normal","Normal","ST","Normal","ST","Normal","Normal","Normal","ST","ST","Normal","Normal","Normal","ST","ST","ST","LVH","Normal","LVH","Normal","Normal","Normal","ST","ST","ST","Normal","LVH","ST","ST","Normal","ST","LVH","Normal","ST","ST","ST","Normal","Normal","Normal","Normal","ST","ST","ST","Normal","ST","ST","ST","ST","Normal","Normal","ST","Normal","ST","LVH","ST","Normal","ST","Normal","LVH","ST","LVH","Normal","LVH","ST","ST","Normal","ST","Normal","LVH","LVH","LVH","LVH","LVH","LVH","Normal","Normal","Normal","Normal","ST","ST","ST","Normal","ST","ST","LVH","LVH","Normal","LVH","LVH","LVH","LVH","Normal","Normal","LVH","LVH","LVH","LVH","LVH","LVH","LVH","LVH","LVH","Normal","Normal","LVH","LVH","Normal","LVH","Normal","LVH","Normal","LVH","Normal","LVH","LVH","Normal","Normal","LVH","Normal","Normal","LVH","LVH","LVH","LVH","LVH","Normal","LVH","Normal","Normal","LVH","LVH","LVH","LVH","Normal","ST","LVH","LVH","LVH","LVH","Normal","LVH","Normal","Normal","Normal","Normal","LVH","LVH","LVH","LVH","Normal","LVH","Normal","Normal","LVH","LVH","Normal","Normal","Normal","Normal","LVH","LVH","LVH","LVH","Normal","Normal","Normal","Normal","LVH","LVH","LVH","Normal","Normal","LVH","LVH","LVH","LVH","Normal","LVH","Normal","LVH","LVH","Normal","LVH","LVH","Normal","LVH","LVH","Normal","Normal","LVH","LVH","LVH","LVH","Normal","LVH","LVH","LVH","Normal","LVH","Normal","Normal","LVH","LVH","Normal","LVH","LVH","Normal","Normal","LVH","Normal","ST","Normal","ST","LVH","LVH","Normal","LVH","Normal","LVH","Normal","Normal","Normal","Normal","LVH"],"xaxis":"x","yaxis":"y"}],                        {"barmode":"relative","height":400,"legend":{"title":{"text":"HeartDisease"},"tracegroupgap":0},"margin":{"t":60},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"width":400,"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"RestingECG"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('8c32b8a4-c7c6-4781-a765-4c05dae3d97a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- RestingECG: resting electrocardiogram results don't differ much.
- Person with ST: having ST-T wave abnormality is more likely have a heart disease than the others.

### **ExerciseAngina and Heart Disease**


```python
df.groupby('ExerciseAngina')['HeartDisease'].mean().sort_values(ascending=False)
```




    ExerciseAngina
    Y    0.851752
    N    0.351005
    Name: HeartDisease, dtype: float64




```python
fig = px.histogram(df, x="ExerciseAngina", color="HeartDisease",width=400, height=400)
fig.show()
```


<div>                            <div id="f605e930-c051-4f58-a8fd-19c93638c813" class="plotly-graph-div" style="height:400px; width:400px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f605e930-c051-4f58-a8fd-19c93638c813")) {                    Plotly.newPlot(                        "f605e930-c051-4f58-a8fd-19c93638c813",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=0<br>ExerciseAngina=%{x}<br>count=%{y}<extra></extra>","legendgroup":"0","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"0","offsetgroup":"0","orientation":"v","showlegend":true,"type":"histogram","x":["N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","Y","N","N","N","Y","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","Y","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","Y","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","Y","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","Y","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","Y","N","Y","N","N","N","Y","N","Y","Y","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","Y","N","N","N","N","Y","N","N","Y","Y","N","N","N","N","N","Y","Y","N","N","N","N","Y","Y","N","N","Y","N","N","N","Y","Y","N","N","N","Y","N","Y","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","Y","N","N","N","Y","N","N","N","N","Y","N","N","N","N","Y","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","Y","N","Y","Y","N","N","N","N","N","N","N","N","N","N","Y","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N"],"xaxis":"x","yaxis":"y"},{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=1<br>ExerciseAngina=%{x}<br>count=%{y}<extra></extra>","legendgroup":"1","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"1","offsetgroup":"1","orientation":"v","showlegend":true,"type":"histogram","x":["N","Y","Y","Y","Y","N","N","N","Y","N","N","N","Y","Y","Y","N","Y","Y","Y","N","Y","Y","Y","Y","N","Y","N","N","N","Y","Y","Y","N","Y","Y","Y","N","Y","N","Y","Y","N","Y","N","N","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","Y","N","Y","Y","Y","N","Y","Y","N","Y","Y","Y","N","N","N","Y","Y","N","N","Y","N","Y","Y","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","N","Y","Y","N","Y","Y","Y","Y","N","Y","N","Y","N","Y","Y","Y","N","N","N","N","Y","N","Y","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","Y","N","Y","Y","Y","Y","N","N","N","N","N","N","N","N","N","Y","N","N","N","N","N","N","Y","Y","N","Y","Y","Y","N","N","Y","N","Y","N","Y","Y","N","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","Y","N","Y","Y","Y","Y","Y","N","N","Y","Y","Y","N","N","N","Y","N","Y","Y","N","N","Y","Y","N","N","N","N","Y","Y","Y","Y","Y","N","Y","N","N","Y","N","Y","N","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","Y","N","N","Y","N","N","Y","Y","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","N","N","Y","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","Y","Y","Y","N","N","Y","Y","Y","Y","Y","Y","N","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","N","Y","Y","N","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","Y","Y","N","Y","Y","N","N","N","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","N","Y","Y","N","Y","N","Y","Y","N","Y","N","Y","Y","Y","Y","Y","Y","N","N","N","Y","N","N","Y","Y","N","N","N","Y","Y","Y","N","N","Y","Y","Y","Y","N","N","N","N","N","Y","N","Y","Y","Y","N","Y","Y","N","N","Y","Y","N","Y","Y","N","N","Y","Y","N","Y","N","N","N","N","N","N","Y","Y","N","N","Y","Y","N","Y","Y","Y","Y","Y","Y","Y","Y","N","N","N","Y","N","Y","N","Y","Y","Y","Y","Y","N","N","Y","N","Y","N","N","Y","N","Y","Y","Y","Y","Y","N","Y","Y","Y","Y","N","Y","Y","N","Y","N","Y","Y","N","Y","N","Y","Y","N","Y","N","N","Y","N","N","Y","N","Y","N","N","Y","Y","Y","N","N","Y","Y","N","N","Y","N","Y","Y","Y","N","Y","N","N","Y","N"],"xaxis":"x","yaxis":"y"}],                        {"barmode":"relative","height":400,"legend":{"title":{"text":"HeartDisease"},"tracegroupgap":0},"margin":{"t":60},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"width":400,"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"ExerciseAngina"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('f605e930-c051-4f58-a8fd-19c93638c813');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- ExerciseAngina: exercise-induced angina with 'Yes' almost 2.4 times more likley have a heart disaese than exercise-induced angina with 'No'

### **ST_Slope and Heart Disease**


```python
df.groupby('ST_Slope')['HeartDisease'].mean().sort_values(ascending=False)
```




    ST_Slope
    Flat    0.828261
    Down    0.777778
    Up      0.197468
    Name: HeartDisease, dtype: float64




```python
fig = px.histogram(df, x="ST_Slope", color="HeartDisease",width=400, height=400)
fig.show()
```


<div>                            <div id="c9062182-0833-479d-80e4-9d7910b2837d" class="plotly-graph-div" style="height:400px; width:400px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c9062182-0833-479d-80e4-9d7910b2837d")) {                    Plotly.newPlot(                        "c9062182-0833-479d-80e4-9d7910b2837d",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=0<br>ST_Slope=%{x}<br>count=%{y}<extra></extra>","legendgroup":"0","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"0","offsetgroup":"0","orientation":"v","showlegend":true,"type":"histogram","x":["Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Flat","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Flat","Up","Up","Flat","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Flat","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Down","Up","Down","Up","Up","Flat","Flat","Up","Flat","Up","Flat","Up","Up","Down","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Down","Up","Up","Up","Up","Flat","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Down","Up","Up","Flat","Flat","Up","Flat","Up","Up","Up","Flat","Up","Up","Up","Up","Flat","Flat","Up","Up","Flat","Up","Up","Up","Flat","Flat","Up","Flat","Flat","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Down","Up","Flat","Up","Up","Flat","Up","Flat","Up","Flat","Up","Flat","Flat","Up","Up","Down","Up","Up","Flat","Up","Up","Flat","Up","Flat","Flat","Up","Up","Up","Flat","Down","Up","Down","Up","Flat","Up","Up","Up","Down","Up","Up","Up","Up","Up","Up","Up","Flat","Up","Flat","Up","Up","Up","Flat","Up","Up","Up","Up","Down","Flat","Flat","Flat","Up","Up","Down","Flat","Up","Up","Up","Flat","Up","Up","Up","Flat","Flat","Flat","Up","Up","Flat","Up","Flat","Down","Flat","Flat","Up","Up","Up","Up","Flat","Up","Up","Up","Flat","Up","Flat","Up","Flat","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Up","Flat","Up","Up","Up","Up","Up","Flat","Flat","Up","Up","Flat","Flat","Flat","Flat","Up","Up","Flat","Up","Down","Up","Up","Up"],"xaxis":"x","yaxis":"y"},{"alignmentgroup":"True","bingroup":"x","hovertemplate":"HeartDisease=1<br>ST_Slope=%{x}<br>count=%{y}<extra></extra>","legendgroup":"1","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"1","offsetgroup":"1","orientation":"v","showlegend":true,"type":"histogram","x":["Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Up","Flat","Up","Flat","Up","Flat","Up","Down","Flat","Up","Flat","Up","Down","Up","Up","Up","Up","Flat","Up","Flat","Up","Flat","Flat","Up","Down","Flat","Down","Up","Flat","Down","Up","Up","Up","Up","Flat","Up","Up","Down","Down","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Down","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Up","Flat","Up","Flat","Down","Up","Flat","Flat","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Up","Flat","Up","Flat","Flat","Flat","Flat","Flat","Down","Flat","Up","Flat","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Down","Flat","Up","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Flat","Down","Up","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Down","Flat","Flat","Down","Flat","Flat","Down","Flat","Flat","Flat","Up","Flat","Flat","Down","Up","Down","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Down","Flat","Flat","Down","Flat","Down","Flat","Down","Up","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Flat","Flat","Down","Flat","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Up","Flat","Flat","Down","Flat","Flat","Flat","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Down","Flat","Flat","Flat","Down","Flat","Up","Up","Up","Up","Flat","Up","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Up","Flat","Flat","Flat","Flat","Up","Flat","Up","Flat","Flat","Up","Flat","Up","Flat","Flat","Flat","Flat","Flat","Up","Flat","Up","Up","Down","Down","Flat","Flat","Flat","Flat","Up","Flat","Up","Down","Flat","Flat","Flat","Up","Flat","Up","Flat","Up","Down","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Down","Up","Up","Up","Flat","Up","Flat","Flat","Flat","Flat","Flat","Flat","Up","Flat","Flat","Flat","Flat","Up","Down","Flat","Flat","Down","Up","Up","Flat","Flat","Flat","Up","Flat","Up","Flat","Flat","Down","Flat","Flat","Flat","Up","Up","Flat","Flat","Up","Down","Flat","Flat","Down","Up","Flat","Flat","Flat","Flat","Flat","Flat","Flat"],"xaxis":"x","yaxis":"y"}],                        {"barmode":"relative","height":400,"legend":{"title":{"text":"HeartDisease"},"tracegroupgap":0},"margin":{"t":60},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"width":400,"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"ST_Slope"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c9062182-0833-479d-80e4-9d7910b2837d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


- ST_Slope: the slope of the peak exercise ST segment has differences.
-  ST_Slope Up significantly less likely has heart disease than the other two segment.

### Overall Insights from the Exploratory Data Analysis

- Target variable has close to balanced data.
- Numerical features have weak correlation with the target variable.
- Oldpeak (depression related number) has a positive correlation with the heart disease.
- Maximum heart rate has negative correlation with the heart disease.
- Interestingly cholesterol has negative correlation with the heart disease.
- Based on the gender; Men are almost 2.44 times more likely have a heart disease than women.
- We can observe clear differences among the chest pain type.
- Person with ASY: Asymptomatic chest pain  has almost 6 times more likely have a heart disease than person with ATA Atypical Angina chest pain.
- RestingECG: resting electrocardiogram results don't differ much.
- Person with ST: having ST-T wave abnormality is more likely have a heart disease than the others.
- ExerciseAngina: exercise-induced angina with 'Yes' almost 2.4 times more likley have a heart disaese than exercise-induced angina with 'No'
- ST_Slope: the slope of the peak exercise ST segment has differences.
- ST_Slope Up significantly less likely has heart disease than the other two segment.




<a id="6"></a>
<font color="lightseagreen" size=+2.5><b>MODEL SELECTION</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

- We'll use dummy classifier model as a base model
-  And then we will use Logistic & Linear Discriminant & KNeighbors and Support Vector Machine models with and without scaler.
- And then we will use ensemble models, Adaboost, Randomforest, Gradient Boosting and Extra Trees
- We will see famous trio: XGBoost,LightGBM & Catboost
- Finally we will look in detail to hyperparameter tuning for Catboost
- Let's start.

<a id="7"></a>
<font color="lightseagreen" size=+1.5><b>Baseline Model</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ohe= OneHotEncoder()
ct= make_column_transformer((ohe,categorical),remainder='passthrough')  


model = DummyClassifier(strategy='constant', constant=1)
pipe = make_pipeline(ct, model)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
accuracy.append(round(accuracy_score(y_test, y_pred),4))
print (f'model : {model} and  accuracy score is : {round(accuracy_score(y_test, y_pred),4)}')

model_names = ['DummyClassifier']
dummy_result_df = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
dummy_result_df
```

    model : DummyClassifier(constant=1, strategy='constant') and  accuracy score is : 0.5942
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DummyClassifier</th>
      <td>0.5942</td>
    </tr>
  </tbody>
</table>
</div>



<a id="8"></a>
<font color="lightseagreen" size=+1.5><b>Logistic & Linear Discriminant & SVC & KNN</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ohe= OneHotEncoder()
ct= make_column_transformer((ohe,categorical),remainder='passthrough')  


lr = LogisticRegression(solver='liblinear')
lda= LinearDiscriminantAnalysis()
svm = SVC(gamma='scale')
knn = KNeighborsClassifier()

models = [lr,lda,svm,knn]

for model in models: 
    pipe = make_pipeline(ct, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy.append(round(accuracy_score(y_test, y_pred),4))
    print (f'model : {model} and  accuracy score is : {round(accuracy_score(y_test, y_pred),4)}')

model_names = ['Logistic','LinearDiscriminant','SVM','KNeighbors']
result_df1 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df1
```

    model : LogisticRegression(solver='liblinear') and  accuracy score is : 0.8841
    model : LinearDiscriminantAnalysis() and  accuracy score is : 0.8696
    model : SVC() and  accuracy score is : 0.7246
    model : KNeighborsClassifier() and  accuracy score is : 0.7174
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic</th>
      <td>0.8841</td>
    </tr>
    <tr>
      <th>LinearDiscriminant</th>
      <td>0.8696</td>
    </tr>
    <tr>
      <th>SVM</th>
      <td>0.7246</td>
    </tr>
    <tr>
      <th>KNeighbors</th>
      <td>0.7174</td>
    </tr>
  </tbody>
</table>
</div>



<a id="9"></a>
<font color="lightseagreen" size=+1.5><b> Logistic & Linear Discriminant & SVC & KNN with Scaler</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ohe= OneHotEncoder()
s= StandardScaler()
ct1= make_column_transformer((ohe,categorical),(s,numerical))  


lr = LogisticRegression(solver='liblinear')
lda= LinearDiscriminantAnalysis()
svm = SVC(gamma='scale')
knn = KNeighborsClassifier()

models = [lr,lda,svm,knn]

for model in models: 
    pipe = make_pipeline(ct1, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy.append(round(accuracy_score(y_test, y_pred),4))
    print (f'model : {model} and  accuracy score is : {round(accuracy_score(y_test, y_pred),4)}')

model_names = ['Logistic_scl','LinearDiscriminant_scl','SVM_scl','KNeighbors_scl']
result_df2 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df2
```

    model : LogisticRegression(solver='liblinear') and  accuracy score is : 0.8804
    model : LinearDiscriminantAnalysis() and  accuracy score is : 0.8696
    model : SVC() and  accuracy score is : 0.8841
    model : KNeighborsClassifier() and  accuracy score is : 0.8841
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic_scl</th>
      <td>0.8804</td>
    </tr>
    <tr>
      <th>LinearDiscriminant_scl</th>
      <td>0.8696</td>
    </tr>
    <tr>
      <th>SVM_scl</th>
      <td>0.8841</td>
    </tr>
    <tr>
      <th>KNeighbors_scl</th>
      <td>0.8841</td>
    </tr>
  </tbody>
</table>
</div>



- As expected, with scaler, both KNN and SVM did a better job with the scaler than their previous performances.

- Let's see how ensemble models do with the problem at hand.

<a id="10"></a>
<font color="lightseagreen" size=+1.5><b>Ensemble Models (AdaBoost & Gradient Boosting & Random Forest & Extra Trees)</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ohe= OneHotEncoder()
ct= make_column_transformer((ohe,categorical),remainder='passthrough')  

ada = AdaBoostClassifier(random_state=0)
gb = GradientBoostingClassifier(random_state=0)
rf = RandomForestClassifier(random_state=0)
et=  ExtraTreesClassifier(random_state=0)



models = [ada,gb,rf,et]

for model in models: 
    pipe = make_pipeline(ct, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy.append(round(accuracy_score(y_test, y_pred),4))
    print (f'model : {model} and  accuracy score is : {round(accuracy_score(y_test, y_pred),4)}')

model_names = ['Ada','Gradient','Random','ExtraTree']
result_df3 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df3
```

    model : AdaBoostClassifier(random_state=0) and  accuracy score is : 0.8659
    model : GradientBoostingClassifier(random_state=0) and  accuracy score is : 0.8768
    model : RandomForestClassifier(random_state=0) and  accuracy score is : 0.8877
    model : ExtraTreesClassifier(random_state=0) and  accuracy score is : 0.8804
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ada</th>
      <td>0.8659</td>
    </tr>
    <tr>
      <th>Gradient</th>
      <td>0.8768</td>
    </tr>
    <tr>
      <th>Random</th>
      <td>0.8877</td>
    </tr>
    <tr>
      <th>ExtraTree</th>
      <td>0.8804</td>
    </tr>
  </tbody>
</table>
</div>



- Accuracy scores are very close to each other.
- Both Random Forest and Extra tree got similar accuracy scores.
- Both model can be improved by hyperparameter tuning.

- OK. Let's see the very famous trio:
  - XGBoost
  - Light GBM
  - Catboost

<a id="11"></a>
<font color="lightseagreen" size=+1.5><b>Famous Trio (XGBoost & LightGBM & Catboost)</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

- I'll use Catboost alone by using its capability to handle categorical variables without doing any preprocessing.
- Let's first look at the XGBoost and LightGBM


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ohe= OneHotEncoder()
ct= make_column_transformer((ohe,categorical),remainder='passthrough')  

xgbc = XGBClassifier(random_state=0)
lgbmc=LGBMClassifier(random_state=0)


models = [xgbc,lgbmc]

for model in models: 
    pipe = make_pipeline(ct, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy.append(round(accuracy_score(y_test, y_pred),4))

model_names = ['XGBoost','LightGBM']
result_df4 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df4
```

    [19:00:23] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>XGBoost</th>
      <td>0.8297</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>0.8732</td>
    </tr>
  </tbody>
</table>
</div>



### Catboost
<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

**Purpose**: 
   
   Training and applying models for the classification problems. Provides compatibility with the scikit-learn tools.

**The default optimized objective depends on various conditions**:

**Logloss** — The target has only two different values or the target_border parameter is not None.

**MultiClass** — The target has more than two different values and the border_count parameter is None.

Reference: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier



```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
categorical_features_indices = np.where(X.dtypes != np.float)[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(verbose=False,random_state=0)

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
y_pred = model.predict(X_test)
accuracy.append(round(accuracy_score(y_test, y_pred),4))

model_names = ['Catboost_default']
result_df5 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df5


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Catboost_default</th>
      <td>0.8804</td>
    </tr>
  </tbody>
</table>
</div>



- Let's make some adjustment on the Catboost model to see its' peak performance on the problem.

<a id="13"></a>
<font color="lightseagreen" size=+1.5><b>Catboost HyperParameter Tuning with Optuna</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
def objective(trial):
    X= df.drop('HeartDisease', axis=1)
    y= df['HeartDisease']
    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat_cls = CatBoostClassifier(**param)

    cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices,verbose=0, early_stopping_rounds=100)

    preds = cat_cls.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

    [32m[I 2021-10-02 19:12:27,447][0m A new study created in memory with name: no-name-8a85af81-fb67-4ad4-8c1d-db929eb1a88e[0m
    [32m[I 2021-10-02 19:12:28,033][0m Trial 0 finished with value: 0.8731884057971014 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.02062214221352644, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 3.1851210741447478}. Best is trial 0 with value: 0.8731884057971014.[0m
    [32m[I 2021-10-02 19:12:28,896][0m Trial 1 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.013983406066328996, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.5774351887788851}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:34,377][0m Trial 2 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09765831645072198, 'depth': 11, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.8333340088928841}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:35,245][0m Trial 3 finished with value: 0.8659420289855072 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.017705168173859406, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 2.7773070625987994}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:36,292][0m Trial 4 finished with value: 0.8768115942028986 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0597216018033412, 'depth': 10, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 6.221287350189732}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:36,888][0m Trial 5 finished with value: 0.8804347826086957 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.03313286088798096, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 6.214218993662279}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:37,937][0m Trial 6 finished with value: 0.8913043478260869 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.03917873590693298, 'depth': 8, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:39,171][0m Trial 7 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0565107890004043, 'depth': 9, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.9230815202815822}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:40,369][0m Trial 8 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06810702019592166, 'depth': 11, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:41,590][0m Trial 9 finished with value: 0.8840579710144928 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.06913279972978485, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:42,515][0m Trial 10 finished with value: 0.8840579710144928 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.09062317378318507, 'depth': 2, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.2770292572195538}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:43,539][0m Trial 11 finished with value: 0.8913043478260869 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.03987706672991701, 'depth': 5, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:44,395][0m Trial 12 finished with value: 0.8876811594202898 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.010242421083110395, 'depth': 4, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.4832417145668411}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:45,373][0m Trial 13 finished with value: 0.8913043478260869 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.03808310544957809, 'depth': 2, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:46,318][0m Trial 14 finished with value: 0.8876811594202898 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.02869399479115184, 'depth': 4, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:48,138][0m Trial 15 finished with value: 0.8876811594202898 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.04730411371358435, 'depth': 12, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.5949978944801486}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:49,042][0m Trial 16 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.022468416059991128, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.15232213280120466}. Best is trial 1 with value: 0.8913043478260869.[0m
    [32m[I 2021-10-02 19:12:51,194][0m Trial 17 finished with value: 0.9021739130434783 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07461412258635804, 'depth': 12, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:51,780][0m Trial 18 finished with value: 0.894927536231884 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.08198545719691544, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:52,410][0m Trial 19 finished with value: 0.8913043478260869 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.07662527656731631, 'depth': 2, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:52,977][0m Trial 20 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08328280731507705, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:53,530][0m Trial 21 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08508780802575587, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:54,102][0m Trial 22 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08095158204886076, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:54,765][0m Trial 23 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07327188523305336, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:55,457][0m Trial 24 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06916630264380479, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:56,021][0m Trial 25 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06198334749721593, 'depth': 4, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:56,570][0m Trial 26 finished with value: 0.8840579710144928 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07635077754433989, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:57,377][0m Trial 27 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06798800183819854, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:12:58,089][0m Trial 28 finished with value: 0.8731884057971014 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07359994328714647, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:01,401][0m Trial 29 finished with value: 0.8768115942028986 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09534554846064268, 'depth': 12, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 9.578141164655737}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:02,284][0m Trial 30 finished with value: 0.8768115942028986 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0876631441536144, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:02,851][0m Trial 31 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08092877492277986, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:03,422][0m Trial 32 finished with value: 0.8913043478260869 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.08449263059224864, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:04,052][0m Trial 33 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.050409878046603504, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:04,851][0m Trial 34 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.04639230630170035, 'depth': 2, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:05,762][0m Trial 35 finished with value: 0.8804347826086957 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09080806773967096, 'depth': 9, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.12623630091202198}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:06,403][0m Trial 36 finished with value: 0.8985507246376812 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05034040335240657, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:07,163][0m Trial 37 finished with value: 0.8695652173913043 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0487187204150675, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 9.727188581327594}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:07,709][0m Trial 38 finished with value: 0.8876811594202898 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.05970485849219709, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:09,020][0m Trial 39 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09771585602635127, 'depth': 10, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:09,638][0m Trial 40 finished with value: 0.8840579710144928 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.06129789877147262, 'depth': 2, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.23056298681848375}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:10,398][0m Trial 41 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06976044215487127, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:11,076][0m Trial 42 finished with value: 0.8913043478260869 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08022423630558739, 'depth': 2, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:11,757][0m Trial 43 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05262832535531862, 'depth': 4, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:12,324][0m Trial 44 finished with value: 0.894927536231884 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06498114805979338, 'depth': 1, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:13,309][0m Trial 45 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05157807914908494, 'depth': 11, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:14,043][0m Trial 46 finished with value: 0.8768115942028986 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0734792814960618, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:14,882][0m Trial 47 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06437195939328424, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7711101266968593}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:15,652][0m Trial 48 finished with value: 0.8804347826086957 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07766706624742202, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    [32m[I 2021-10-02 19:13:16,364][0m Trial 49 finished with value: 0.8876811594202898 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.042550900595807566, 'depth': 4, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 17 with value: 0.9021739130434783.[0m
    

    Number of finished trials: 50
    Best trial:
      Value: 0.9021739130434783
      Params: 
        objective: CrossEntropy
        colsample_bylevel: 0.07461412258635804
        depth: 12
        boosting_type: Plain
        bootstrap_type: MVS
    

**Parameters**:

- **Objective**:  Supported metrics for overfitting detection and best model selection 

- **colsample_bylevel**: this parameter speeds up the training and usually does not affect the quality.

- **depht** : Depth of the tree.


- **boosting_type** : By default, the boosting type is set to for small datasets. This prevents overfitting but it is expensive in terms of computation. Try to set the value of this parameter to  to speed up the training.

- **bootstrap_type** : By default, the method for sampling the weights of objects is set to . The training is performed faster if the method is set and the value for the sample rate for bagging is smaller than 1.


Reference: https://catboost.ai/




- Ok let's use our best model with new parameters.


```python
accuracy =[]
model_names =[]


X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
categorical_features_indices = np.where(X.dtypes != np.float)[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(verbose=False,random_state=0,
                          objective= 'CrossEntropy',
    colsample_bylevel= 0.04292240490294766,
    depth= 10,
    boosting_type= 'Plain',
    bootstrap_type= 'MVS')

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
y_pred = model.predict(X_test)
accuracy.append(round(accuracy_score(y_test, y_pred),4))
print(classification_report(y_test, y_pred))

model_names = ['Catboost_tuned']
result_df6 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df6


```

                  precision    recall  f1-score   support
    
               0       0.88      0.90      0.89       112
               1       0.93      0.91      0.92       164
    
        accuracy                           0.91       276
       macro avg       0.90      0.91      0.91       276
    weighted avg       0.91      0.91      0.91       276
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Catboost_tuned</th>
      <td>0.9094</td>
    </tr>
  </tbody>
</table>
</div>




```python
#X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>668</th>
      <td>63</td>
      <td>F</td>
      <td>ATA</td>
      <td>140</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>179</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>30</th>
      <td>53</td>
      <td>M</td>
      <td>NAP</td>
      <td>145</td>
      <td>518</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>377</th>
      <td>65</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>122</td>
      <td>N</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>535</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>0</td>
      <td>0</td>
      <td>LVH</td>
      <td>122</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>807</th>
      <td>54</td>
      <td>M</td>
      <td>ATA</td>
      <td>108</td>
      <td>309</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>793</th>
      <td>67</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>254</td>
      <td>1</td>
      <td>Normal</td>
      <td>163</td>
      <td>N</td>
      <td>0.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>363</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>148</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>583</th>
      <td>69</td>
      <td>M</td>
      <td>NAP</td>
      <td>142</td>
      <td>271</td>
      <td>0</td>
      <td>LVH</td>
      <td>126</td>
      <td>N</td>
      <td>0.3</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>165</th>
      <td>46</td>
      <td>M</td>
      <td>TA</td>
      <td>140</td>
      <td>272</td>
      <td>1</td>
      <td>Normal</td>
      <td>175</td>
      <td>N</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>483</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>LVH</td>
      <td>106</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>773</th>
      <td>56</td>
      <td>M</td>
      <td>TA</td>
      <td>120</td>
      <td>193</td>
      <td>0</td>
      <td>LVH</td>
      <td>162</td>
      <td>N</td>
      <td>1.9</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>551</th>
      <td>62</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>220</td>
      <td>0</td>
      <td>LVH</td>
      <td>86</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>768</th>
      <td>64</td>
      <td>F</td>
      <td>ASY</td>
      <td>130</td>
      <td>303</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>694</th>
      <td>56</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>Normal</td>
      <td>178</td>
      <td>N</td>
      <td>0.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>718</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>165</td>
      <td>289</td>
      <td>1</td>
      <td>LVH</td>
      <td>124</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>312</th>
      <td>41</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>176</td>
      <td>N</td>
      <td>1.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>713</th>
      <td>64</td>
      <td>F</td>
      <td>NAP</td>
      <td>140</td>
      <td>313</td>
      <td>0</td>
      <td>Normal</td>
      <td>133</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>309</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>95</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>182</td>
      <td>N</td>
      <td>0.7</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>846</th>
      <td>39</td>
      <td>M</td>
      <td>ASY</td>
      <td>118</td>
      <td>219</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>N</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>616</th>
      <td>67</td>
      <td>F</td>
      <td>NAP</td>
      <td>115</td>
      <td>564</td>
      <td>0</td>
      <td>LVH</td>
      <td>160</td>
      <td>N</td>
      <td>1.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>355</th>
      <td>67</td>
      <td>M</td>
      <td>TA</td>
      <td>145</td>
      <td>0</td>
      <td>0</td>
      <td>LVH</td>
      <td>125</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>39</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>150</td>
      <td>227</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>231</th>
      <td>40</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>281</td>
      <td>0</td>
      <td>Normal</td>
      <td>167</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>822</th>
      <td>58</td>
      <td>M</td>
      <td>NAP</td>
      <td>105</td>
      <td>240</td>
      <td>0</td>
      <td>LVH</td>
      <td>154</td>
      <td>Y</td>
      <td>0.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>603</th>
      <td>74</td>
      <td>M</td>
      <td>ASY</td>
      <td>155</td>
      <td>310</td>
      <td>0</td>
      <td>Normal</td>
      <td>112</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>63</th>
      <td>46</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>277</td>
      <td>0</td>
      <td>Normal</td>
      <td>125</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>192</th>
      <td>48</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>245</td>
      <td>0</td>
      <td>Normal</td>
      <td>160</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>481</th>
      <td>69</td>
      <td>M</td>
      <td>NAP</td>
      <td>140</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>118</td>
      <td>N</td>
      <td>2.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>866</th>
      <td>44</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>219</td>
      <td>0</td>
      <td>LVH</td>
      <td>188</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>67</th>
      <td>32</td>
      <td>M</td>
      <td>ATA</td>
      <td>110</td>
      <td>225</td>
      <td>0</td>
      <td>Normal</td>
      <td>184</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>72</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>182</td>
      <td>0</td>
      <td>Normal</td>
      <td>150</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>655</th>
      <td>40</td>
      <td>M</td>
      <td>ASY</td>
      <td>152</td>
      <td>223</td>
      <td>0</td>
      <td>Normal</td>
      <td>181</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>679</th>
      <td>63</td>
      <td>M</td>
      <td>TA</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>LVH</td>
      <td>150</td>
      <td>N</td>
      <td>2.3</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>139</th>
      <td>43</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>247</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>732</th>
      <td>56</td>
      <td>F</td>
      <td>ASY</td>
      <td>200</td>
      <td>288</td>
      <td>1</td>
      <td>LVH</td>
      <td>133</td>
      <td>Y</td>
      <td>4.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>824</th>
      <td>37</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>Normal</td>
      <td>187</td>
      <td>N</td>
      <td>3.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>174</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>266</td>
      <td>0</td>
      <td>Normal</td>
      <td>134</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>896</th>
      <td>47</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>253</td>
      <td>0</td>
      <td>Normal</td>
      <td>179</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>499</th>
      <td>62</td>
      <td>M</td>
      <td>ASY</td>
      <td>135</td>
      <td>297</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>70</th>
      <td>57</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>265</td>
      <td>0</td>
      <td>ST</td>
      <td>145</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>716</th>
      <td>67</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>237</td>
      <td>0</td>
      <td>Normal</td>
      <td>71</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>23</th>
      <td>44</td>
      <td>M</td>
      <td>ATA</td>
      <td>150</td>
      <td>288</td>
      <td>0</td>
      <td>Normal</td>
      <td>150</td>
      <td>Y</td>
      <td>3.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>541</th>
      <td>76</td>
      <td>M</td>
      <td>NAP</td>
      <td>104</td>
      <td>113</td>
      <td>0</td>
      <td>LVH</td>
      <td>120</td>
      <td>N</td>
      <td>3.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>799</th>
      <td>53</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>246</td>
      <td>1</td>
      <td>LVH</td>
      <td>173</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>672</th>
      <td>60</td>
      <td>F</td>
      <td>NAP</td>
      <td>120</td>
      <td>178</td>
      <td>1</td>
      <td>Normal</td>
      <td>96</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>826</th>
      <td>51</td>
      <td>M</td>
      <td>NAP</td>
      <td>125</td>
      <td>245</td>
      <td>1</td>
      <td>LVH</td>
      <td>166</td>
      <td>N</td>
      <td>2.4</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>250</th>
      <td>44</td>
      <td>M</td>
      <td>ASY</td>
      <td>135</td>
      <td>491</td>
      <td>0</td>
      <td>Normal</td>
      <td>135</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>752</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>249</td>
      <td>1</td>
      <td>LVH</td>
      <td>144</td>
      <td>Y</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>350</th>
      <td>53</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>120</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>758</th>
      <td>51</td>
      <td>M</td>
      <td>TA</td>
      <td>125</td>
      <td>213</td>
      <td>0</td>
      <td>LVH</td>
      <td>125</td>
      <td>Y</td>
      <td>1.4</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>759</th>
      <td>54</td>
      <td>M</td>
      <td>ATA</td>
      <td>192</td>
      <td>283</td>
      <td>0</td>
      <td>LVH</td>
      <td>195</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>107</th>
      <td>34</td>
      <td>M</td>
      <td>ATA</td>
      <td>150</td>
      <td>214</td>
      <td>0</td>
      <td>ST</td>
      <td>168</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>445</th>
      <td>55</td>
      <td>M</td>
      <td>NAP</td>
      <td>136</td>
      <td>228</td>
      <td>0</td>
      <td>ST</td>
      <td>124</td>
      <td>Y</td>
      <td>1.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>141</th>
      <td>50</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>341</td>
      <td>0</td>
      <td>ST</td>
      <td>125</td>
      <td>Y</td>
      <td>2.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>650</th>
      <td>48</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>256</td>
      <td>1</td>
      <td>LVH</td>
      <td>150</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>544</th>
      <td>61</td>
      <td>F</td>
      <td>ATA</td>
      <td>140</td>
      <td>298</td>
      <td>1</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>110</th>
      <td>59</td>
      <td>F</td>
      <td>ATA</td>
      <td>130</td>
      <td>188</td>
      <td>0</td>
      <td>Normal</td>
      <td>124</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>593</th>
      <td>64</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>258</td>
      <td>1</td>
      <td>LVH</td>
      <td>130</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>519</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>96</td>
      <td>305</td>
      <td>0</td>
      <td>ST</td>
      <td>121</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>907</th>
      <td>44</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>169</td>
      <td>0</td>
      <td>Normal</td>
      <td>144</td>
      <td>Y</td>
      <td>2.8</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>126</td>
      <td>1</td>
      <td>Normal</td>
      <td>173</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>280</th>
      <td>60</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>246</td>
      <td>0</td>
      <td>LVH</td>
      <td>135</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>136</th>
      <td>43</td>
      <td>F</td>
      <td>ATA</td>
      <td>120</td>
      <td>215</td>
      <td>0</td>
      <td>ST</td>
      <td>175</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>422</th>
      <td>65</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>236</td>
      <td>1</td>
      <td>ST</td>
      <td>105</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>208</th>
      <td>28</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>132</td>
      <td>0</td>
      <td>LVH</td>
      <td>185</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>442</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>128</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>125</td>
      <td>Y</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>86</th>
      <td>65</td>
      <td>M</td>
      <td>ASY</td>
      <td>170</td>
      <td>263</td>
      <td>1</td>
      <td>Normal</td>
      <td>112</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>44</th>
      <td>43</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>175</td>
      <td>0</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>531</th>
      <td>64</td>
      <td>M</td>
      <td>ASY</td>
      <td>143</td>
      <td>306</td>
      <td>1</td>
      <td>ST</td>
      <td>115</td>
      <td>Y</td>
      <td>1.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>913</th>
      <td>45</td>
      <td>M</td>
      <td>TA</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>Normal</td>
      <td>132</td>
      <td>N</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>634</th>
      <td>40</td>
      <td>M</td>
      <td>TA</td>
      <td>140</td>
      <td>199</td>
      <td>0</td>
      <td>Normal</td>
      <td>178</td>
      <td>Y</td>
      <td>1.4</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>290</th>
      <td>48</td>
      <td>M</td>
      <td>NAP</td>
      <td>110</td>
      <td>211</td>
      <td>0</td>
      <td>Normal</td>
      <td>138</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>338</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>0</td>
      <td>1</td>
      <td>LVH</td>
      <td>149</td>
      <td>N</td>
      <td>2.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>357</th>
      <td>53</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>95</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>292</th>
      <td>53</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>182</td>
      <td>0</td>
      <td>Normal</td>
      <td>148</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>227</th>
      <td>38</td>
      <td>M</td>
      <td>ASY</td>
      <td>92</td>
      <td>117</td>
      <td>0</td>
      <td>Normal</td>
      <td>134</td>
      <td>Y</td>
      <td>2.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>591</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>100</td>
      <td>213</td>
      <td>0</td>
      <td>ST</td>
      <td>110</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>425</th>
      <td>60</td>
      <td>M</td>
      <td>ATA</td>
      <td>160</td>
      <td>267</td>
      <td>1</td>
      <td>ST</td>
      <td>157</td>
      <td>N</td>
      <td>0.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>789</th>
      <td>34</td>
      <td>M</td>
      <td>TA</td>
      <td>118</td>
      <td>182</td>
      <td>0</td>
      <td>LVH</td>
      <td>174</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>522</th>
      <td>50</td>
      <td>M</td>
      <td>ASY</td>
      <td>144</td>
      <td>349</td>
      <td>0</td>
      <td>LVH</td>
      <td>120</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>861</th>
      <td>65</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>248</td>
      <td>0</td>
      <td>LVH</td>
      <td>158</td>
      <td>N</td>
      <td>0.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>352</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>100</td>
      <td>Y</td>
      <td>-1.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>493</th>
      <td>51</td>
      <td>M</td>
      <td>NAP</td>
      <td>137</td>
      <td>339</td>
      <td>0</td>
      <td>Normal</td>
      <td>127</td>
      <td>Y</td>
      <td>1.7</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>60</th>
      <td>49</td>
      <td>M</td>
      <td>ATA</td>
      <td>100</td>
      <td>253</td>
      <td>0</td>
      <td>Normal</td>
      <td>174</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>598</th>
      <td>55</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>226</td>
      <td>0</td>
      <td>LVH</td>
      <td>127</td>
      <td>Y</td>
      <td>1.7</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>722</th>
      <td>60</td>
      <td>F</td>
      <td>ASY</td>
      <td>150</td>
      <td>258</td>
      <td>0</td>
      <td>LVH</td>
      <td>157</td>
      <td>N</td>
      <td>2.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>426</th>
      <td>56</td>
      <td>M</td>
      <td>ATA</td>
      <td>126</td>
      <td>166</td>
      <td>0</td>
      <td>ST</td>
      <td>140</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>468</th>
      <td>62</td>
      <td>M</td>
      <td>ASY</td>
      <td>152</td>
      <td>153</td>
      <td>0</td>
      <td>ST</td>
      <td>97</td>
      <td>Y</td>
      <td>1.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>66</th>
      <td>45</td>
      <td>F</td>
      <td>ASY</td>
      <td>132</td>
      <td>297</td>
      <td>0</td>
      <td>Normal</td>
      <td>144</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>332</th>
      <td>38</td>
      <td>M</td>
      <td>NAP</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>179</td>
      <td>N</td>
      <td>-1.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>375</th>
      <td>73</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>121</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>381</th>
      <td>50</td>
      <td>M</td>
      <td>ASY</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>0.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>557</th>
      <td>56</td>
      <td>M</td>
      <td>NAP</td>
      <td>137</td>
      <td>208</td>
      <td>1</td>
      <td>ST</td>
      <td>122</td>
      <td>Y</td>
      <td>1.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>260</th>
      <td>46</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>275</td>
      <td>0</td>
      <td>Normal</td>
      <td>165</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>235</th>
      <td>39</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>200</td>
      <td>0</td>
      <td>Normal</td>
      <td>160</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>218</th>
      <td>55</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>196</td>
      <td>0</td>
      <td>Normal</td>
      <td>150</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>382</th>
      <td>43</td>
      <td>M</td>
      <td>ASY</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>145</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>439</th>
      <td>74</td>
      <td>M</td>
      <td>NAP</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>116</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>762</th>
      <td>40</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>167</td>
      <td>0</td>
      <td>LVH</td>
      <td>114</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>305</th>
      <td>51</td>
      <td>F</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>127</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>137</th>
      <td>39</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>241</td>
      <td>0</td>
      <td>ST</td>
      <td>146</td>
      <td>N</td>
      <td>2.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>538</th>
      <td>54</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>202</td>
      <td>1</td>
      <td>Normal</td>
      <td>112</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>76</th>
      <td>32</td>
      <td>M</td>
      <td>ASY</td>
      <td>118</td>
      <td>529</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>794</th>
      <td>50</td>
      <td>M</td>
      <td>NAP</td>
      <td>129</td>
      <td>196</td>
      <td>0</td>
      <td>Normal</td>
      <td>163</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>436</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>124</td>
      <td>N</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>266</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>298</td>
      <td>0</td>
      <td>Normal</td>
      <td>110</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>215</th>
      <td>30</td>
      <td>F</td>
      <td>TA</td>
      <td>170</td>
      <td>237</td>
      <td>0</td>
      <td>ST</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>334</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>-0.7</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>523</th>
      <td>59</td>
      <td>M</td>
      <td>ASY</td>
      <td>124</td>
      <td>160</td>
      <td>0</td>
      <td>Normal</td>
      <td>117</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>507</th>
      <td>40</td>
      <td>M</td>
      <td>NAP</td>
      <td>106</td>
      <td>240</td>
      <td>0</td>
      <td>Normal</td>
      <td>80</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>209</th>
      <td>54</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>216</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>444</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>100</td>
      <td>0</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>749</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>232</td>
      <td>0</td>
      <td>LVH</td>
      <td>165</td>
      <td>N</td>
      <td>1.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>662</th>
      <td>44</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>197</td>
      <td>0</td>
      <td>LVH</td>
      <td>177</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>198</th>
      <td>53</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>246</td>
      <td>0</td>
      <td>Normal</td>
      <td>116</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>839</th>
      <td>35</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>183</td>
      <td>0</td>
      <td>Normal</td>
      <td>182</td>
      <td>N</td>
      <td>1.4</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>299</th>
      <td>47</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>149</td>
      <td>N</td>
      <td>2.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>597</th>
      <td>55</td>
      <td>M</td>
      <td>NAP</td>
      <td>133</td>
      <td>185</td>
      <td>0</td>
      <td>ST</td>
      <td>136</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>685</th>
      <td>61</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>260</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>Y</td>
      <td>3.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>844</th>
      <td>52</td>
      <td>M</td>
      <td>TA</td>
      <td>118</td>
      <td>186</td>
      <td>0</td>
      <td>LVH</td>
      <td>190</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>497</th>
      <td>61</td>
      <td>M</td>
      <td>ASY</td>
      <td>146</td>
      <td>241</td>
      <td>0</td>
      <td>Normal</td>
      <td>148</td>
      <td>Y</td>
      <td>3.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>316</th>
      <td>57</td>
      <td>M</td>
      <td>NAP</td>
      <td>105</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>148</td>
      <td>N</td>
      <td>0.3</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>294</th>
      <td>32</td>
      <td>M</td>
      <td>TA</td>
      <td>95</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>127</td>
      <td>N</td>
      <td>0.7</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>622</th>
      <td>59</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>239</td>
      <td>0</td>
      <td>LVH</td>
      <td>142</td>
      <td>Y</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>903</th>
      <td>56</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>221</td>
      <td>0</td>
      <td>LVH</td>
      <td>163</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>96</th>
      <td>43</td>
      <td>M</td>
      <td>ATA</td>
      <td>142</td>
      <td>207</td>
      <td>0</td>
      <td>Normal</td>
      <td>138</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>467</th>
      <td>63</td>
      <td>F</td>
      <td>ATA</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>N</td>
      <td>0.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>120</th>
      <td>47</td>
      <td>F</td>
      <td>NAP</td>
      <td>135</td>
      <td>248</td>
      <td>1</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>465</th>
      <td>42</td>
      <td>M</td>
      <td>NAP</td>
      <td>134</td>
      <td>240</td>
      <td>0</td>
      <td>Normal</td>
      <td>160</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>548</th>
      <td>66</td>
      <td>M</td>
      <td>ASY</td>
      <td>112</td>
      <td>261</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>N</td>
      <td>1.5</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>78</th>
      <td>52</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>100</td>
      <td>0</td>
      <td>Normal</td>
      <td>138</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>239</th>
      <td>48</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>193</td>
      <td>0</td>
      <td>Normal</td>
      <td>102</td>
      <td>Y</td>
      <td>3.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>210</th>
      <td>48</td>
      <td>M</td>
      <td>ASY</td>
      <td>106</td>
      <td>263</td>
      <td>1</td>
      <td>Normal</td>
      <td>110</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>652</th>
      <td>59</td>
      <td>M</td>
      <td>TA</td>
      <td>160</td>
      <td>273</td>
      <td>0</td>
      <td>LVH</td>
      <td>125</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>412</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>103</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>590</th>
      <td>63</td>
      <td>M</td>
      <td>ATA</td>
      <td>136</td>
      <td>165</td>
      <td>0</td>
      <td>ST</td>
      <td>133</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>513</th>
      <td>62</td>
      <td>M</td>
      <td>TA</td>
      <td>112</td>
      <td>258</td>
      <td>0</td>
      <td>ST</td>
      <td>150</td>
      <td>Y</td>
      <td>1.3</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>792</th>
      <td>46</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>231</td>
      <td>0</td>
      <td>Normal</td>
      <td>147</td>
      <td>N</td>
      <td>3.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>65</th>
      <td>37</td>
      <td>F</td>
      <td>ATA</td>
      <td>120</td>
      <td>260</td>
      <td>0</td>
      <td>Normal</td>
      <td>130</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>319</th>
      <td>68</td>
      <td>M</td>
      <td>ASY</td>
      <td>145</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>136</td>
      <td>N</td>
      <td>1.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>286</th>
      <td>59</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>169</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>611</th>
      <td>62</td>
      <td>M</td>
      <td>TA</td>
      <td>135</td>
      <td>139</td>
      <td>0</td>
      <td>ST</td>
      <td>137</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>328</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>95</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>82</td>
      <td>Y</td>
      <td>0.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>744</th>
      <td>60</td>
      <td>M</td>
      <td>ASY</td>
      <td>117</td>
      <td>230</td>
      <td>1</td>
      <td>Normal</td>
      <td>160</td>
      <td>Y</td>
      <td>1.4</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>429</th>
      <td>63</td>
      <td>M</td>
      <td>NAP</td>
      <td>133</td>
      <td>0</td>
      <td>0</td>
      <td>LVH</td>
      <td>120</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>314</th>
      <td>53</td>
      <td>M</td>
      <td>ASY</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>141</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>49</th>
      <td>41</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>667</th>
      <td>65</td>
      <td>F</td>
      <td>NAP</td>
      <td>140</td>
      <td>417</td>
      <td>1</td>
      <td>LVH</td>
      <td>157</td>
      <td>N</td>
      <td>0.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>168</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>135</td>
      <td>222</td>
      <td>0</td>
      <td>Normal</td>
      <td>100</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>816</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>300</td>
      <td>0</td>
      <td>LVH</td>
      <td>171</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>881</th>
      <td>44</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>263</td>
      <td>0</td>
      <td>Normal</td>
      <td>173</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>912</th>
      <td>57</td>
      <td>F</td>
      <td>ASY</td>
      <td>140</td>
      <td>241</td>
      <td>0</td>
      <td>Normal</td>
      <td>123</td>
      <td>Y</td>
      <td>0.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>572</th>
      <td>64</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>193</td>
      <td>0</td>
      <td>ST</td>
      <td>135</td>
      <td>Y</td>
      <td>0.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>714</th>
      <td>50</td>
      <td>F</td>
      <td>ATA</td>
      <td>120</td>
      <td>244</td>
      <td>0</td>
      <td>Normal</td>
      <td>162</td>
      <td>N</td>
      <td>1.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>33</th>
      <td>41</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>172</td>
      <td>0</td>
      <td>ST</td>
      <td>130</td>
      <td>N</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>31</th>
      <td>56</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>167</td>
      <td>0</td>
      <td>Normal</td>
      <td>114</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>684</th>
      <td>47</td>
      <td>M</td>
      <td>NAP</td>
      <td>108</td>
      <td>243</td>
      <td>0</td>
      <td>Normal</td>
      <td>152</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>417</th>
      <td>44</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>209</td>
      <td>0</td>
      <td>ST</td>
      <td>127</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>525</th>
      <td>45</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>Normal</td>
      <td>144</td>
      <td>N</td>
      <td>0.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>247</th>
      <td>48</td>
      <td>M</td>
      <td>ASY</td>
      <td>122</td>
      <td>275</td>
      <td>1</td>
      <td>ST</td>
      <td>150</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>109</th>
      <td>39</td>
      <td>M</td>
      <td>ATA</td>
      <td>190</td>
      <td>241</td>
      <td>0</td>
      <td>Normal</td>
      <td>106</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>77</th>
      <td>35</td>
      <td>F</td>
      <td>ASY</td>
      <td>140</td>
      <td>167</td>
      <td>0</td>
      <td>Normal</td>
      <td>150</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>346</th>
      <td>59</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>115</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>213</th>
      <td>56</td>
      <td>F</td>
      <td>NAP</td>
      <td>130</td>
      <td>219</td>
      <td>0</td>
      <td>ST</td>
      <td>164</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>82</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>223</td>
      <td>0</td>
      <td>Normal</td>
      <td>115</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>298</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>92</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>331</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>115</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>82</td>
      <td>N</td>
      <td>-1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>432</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>170</td>
      <td>177</td>
      <td>0</td>
      <td>Normal</td>
      <td>84</td>
      <td>Y</td>
      <td>2.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>868</th>
      <td>51</td>
      <td>M</td>
      <td>NAP</td>
      <td>110</td>
      <td>175</td>
      <td>0</td>
      <td>Normal</td>
      <td>123</td>
      <td>N</td>
      <td>0.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>705</th>
      <td>59</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>221</td>
      <td>0</td>
      <td>Normal</td>
      <td>164</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>532</th>
      <td>55</td>
      <td>M</td>
      <td>ASY</td>
      <td>116</td>
      <td>186</td>
      <td>1</td>
      <td>ST</td>
      <td>102</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>599</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>203</td>
      <td>1</td>
      <td>Normal</td>
      <td>98</td>
      <td>N</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>339</td>
      <td>0</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>54</th>
      <td>52</td>
      <td>F</td>
      <td>ASY</td>
      <td>130</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>265</th>
      <td>54</td>
      <td>M</td>
      <td>ATA</td>
      <td>160</td>
      <td>305</td>
      <td>0</td>
      <td>Normal</td>
      <td>175</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>478</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>311</td>
      <td>1</td>
      <td>ST</td>
      <td>148</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>333</th>
      <td>40</td>
      <td>M</td>
      <td>ASY</td>
      <td>95</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>144</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>787</th>
      <td>67</td>
      <td>M</td>
      <td>ASY</td>
      <td>100</td>
      <td>299</td>
      <td>0</td>
      <td>LVH</td>
      <td>125</td>
      <td>Y</td>
      <td>0.9</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>97</th>
      <td>39</td>
      <td>M</td>
      <td>NAP</td>
      <td>160</td>
      <td>147</td>
      <td>1</td>
      <td>Normal</td>
      <td>160</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>211</th>
      <td>50</td>
      <td>F</td>
      <td>NAP</td>
      <td>140</td>
      <td>288</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>745</th>
      <td>63</td>
      <td>F</td>
      <td>ASY</td>
      <td>108</td>
      <td>269</td>
      <td>0</td>
      <td>Normal</td>
      <td>169</td>
      <td>Y</td>
      <td>1.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>584</th>
      <td>64</td>
      <td>M</td>
      <td>ASY</td>
      <td>141</td>
      <td>244</td>
      <td>1</td>
      <td>ST</td>
      <td>116</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>878</th>
      <td>49</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>266</td>
      <td>0</td>
      <td>Normal</td>
      <td>171</td>
      <td>N</td>
      <td>0.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>451</th>
      <td>64</td>
      <td>M</td>
      <td>ASY</td>
      <td>144</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>122</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>795</th>
      <td>42</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>240</td>
      <td>1</td>
      <td>Normal</td>
      <td>194</td>
      <td>N</td>
      <td>0.8</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>25</th>
      <td>36</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>209</td>
      <td>0</td>
      <td>Normal</td>
      <td>178</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>84</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>213</td>
      <td>1</td>
      <td>Normal</td>
      <td>125</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>10</th>
      <td>37</td>
      <td>F</td>
      <td>NAP</td>
      <td>130</td>
      <td>211</td>
      <td>0</td>
      <td>Normal</td>
      <td>142</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>344</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>104</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>254</th>
      <td>55</td>
      <td>M</td>
      <td>ASY</td>
      <td>145</td>
      <td>248</td>
      <td>0</td>
      <td>Normal</td>
      <td>96</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>398</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>165</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>122</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>244</th>
      <td>48</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>268</td>
      <td>0</td>
      <td>Normal</td>
      <td>103</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>621</th>
      <td>56</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>256</td>
      <td>1</td>
      <td>LVH</td>
      <td>142</td>
      <td>Y</td>
      <td>0.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>118</th>
      <td>35</td>
      <td>F</td>
      <td>TA</td>
      <td>120</td>
      <td>160</td>
      <td>0</td>
      <td>ST</td>
      <td>185</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>300</th>
      <td>60</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>149</td>
      <td>N</td>
      <td>0.4</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>326</th>
      <td>45</td>
      <td>M</td>
      <td>NAP</td>
      <td>110</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>138</td>
      <td>N</td>
      <td>-0.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>29</th>
      <td>51</td>
      <td>M</td>
      <td>ATA</td>
      <td>125</td>
      <td>188</td>
      <td>0</td>
      <td>Normal</td>
      <td>145</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>55</th>
      <td>51</td>
      <td>F</td>
      <td>ATA</td>
      <td>160</td>
      <td>194</td>
      <td>0</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>430</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>128</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>148</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>390</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>60</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>911</th>
      <td>59</td>
      <td>M</td>
      <td>ASY</td>
      <td>164</td>
      <td>176</td>
      <td>1</td>
      <td>LVH</td>
      <td>90</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>916</th>
      <td>57</td>
      <td>F</td>
      <td>ATA</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>LVH</td>
      <td>174</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>196</th>
      <td>49</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>297</td>
      <td>0</td>
      <td>Normal</td>
      <td>132</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>404</th>
      <td>47</td>
      <td>M</td>
      <td>NAP</td>
      <td>110</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>81</th>
      <td>54</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>238</td>
      <td>0</td>
      <td>Normal</td>
      <td>154</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>464</th>
      <td>59</td>
      <td>M</td>
      <td>NAP</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>128</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>689</th>
      <td>67</td>
      <td>F</td>
      <td>ASY</td>
      <td>106</td>
      <td>223</td>
      <td>0</td>
      <td>Normal</td>
      <td>142</td>
      <td>N</td>
      <td>0.3</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>568</th>
      <td>38</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>105</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>281</th>
      <td>49</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>222</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>803</th>
      <td>62</td>
      <td>F</td>
      <td>ASY</td>
      <td>140</td>
      <td>394</td>
      <td>0</td>
      <td>LVH</td>
      <td>157</td>
      <td>N</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>204</th>
      <td>56</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>184</td>
      <td>0</td>
      <td>Normal</td>
      <td>100</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>554</th>
      <td>53</td>
      <td>M</td>
      <td>NAP</td>
      <td>155</td>
      <td>175</td>
      <td>1</td>
      <td>ST</td>
      <td>160</td>
      <td>N</td>
      <td>0.3</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>658</th>
      <td>46</td>
      <td>M</td>
      <td>ATA</td>
      <td>101</td>
      <td>197</td>
      <td>1</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>296</th>
      <td>50</td>
      <td>M</td>
      <td>ASY</td>
      <td>145</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>139</td>
      <td>Y</td>
      <td>0.7</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>858</th>
      <td>62</td>
      <td>F</td>
      <td>ASY</td>
      <td>140</td>
      <td>268</td>
      <td>0</td>
      <td>LVH</td>
      <td>160</td>
      <td>N</td>
      <td>3.6</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>547</th>
      <td>61</td>
      <td>M</td>
      <td>TA</td>
      <td>142</td>
      <td>200</td>
      <td>1</td>
      <td>ST</td>
      <td>100</td>
      <td>N</td>
      <td>1.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>368</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>783</th>
      <td>45</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>236</td>
      <td>0</td>
      <td>LVH</td>
      <td>152</td>
      <td>Y</td>
      <td>0.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>829</th>
      <td>29</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>LVH</td>
      <td>202</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>731</th>
      <td>46</td>
      <td>M</td>
      <td>ASY</td>
      <td>120</td>
      <td>249</td>
      <td>0</td>
      <td>LVH</td>
      <td>144</td>
      <td>N</td>
      <td>0.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>882</th>
      <td>56</td>
      <td>F</td>
      <td>ATA</td>
      <td>140</td>
      <td>294</td>
      <td>0</td>
      <td>LVH</td>
      <td>153</td>
      <td>N</td>
      <td>1.3</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>59</th>
      <td>52</td>
      <td>M</td>
      <td>ASY</td>
      <td>112</td>
      <td>342</td>
      <td>0</td>
      <td>ST</td>
      <td>96</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>323</th>
      <td>62</td>
      <td>M</td>
      <td>ASY</td>
      <td>115</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>128</td>
      <td>Y</td>
      <td>2.5</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>862</th>
      <td>65</td>
      <td>F</td>
      <td>NAP</td>
      <td>155</td>
      <td>269</td>
      <td>0</td>
      <td>Normal</td>
      <td>148</td>
      <td>N</td>
      <td>0.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>380</th>
      <td>60</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>99</td>
      <td>Y</td>
      <td>0.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>259</th>
      <td>55</td>
      <td>F</td>
      <td>ATA</td>
      <td>122</td>
      <td>320</td>
      <td>0</td>
      <td>Normal</td>
      <td>155</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>490</th>
      <td>72</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>102</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>302</th>
      <td>53</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>120</td>
      <td>N</td>
      <td>1.5</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54</td>
      <td>M</td>
      <td>ATA</td>
      <td>110</td>
      <td>208</td>
      <td>0</td>
      <td>Normal</td>
      <td>142</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>892</th>
      <td>39</td>
      <td>F</td>
      <td>NAP</td>
      <td>138</td>
      <td>220</td>
      <td>0</td>
      <td>Normal</td>
      <td>152</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>155</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>155</td>
      <td>342</td>
      <td>1</td>
      <td>Normal</td>
      <td>150</td>
      <td>Y</td>
      <td>3.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>362</th>
      <td>56</td>
      <td>M</td>
      <td>NAP</td>
      <td>155</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>99</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>869</th>
      <td>59</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>212</td>
      <td>1</td>
      <td>Normal</td>
      <td>157</td>
      <td>N</td>
      <td>1.6</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>101</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>179</td>
      <td>0</td>
      <td>Normal</td>
      <td>100</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>199</th>
      <td>57</td>
      <td>F</td>
      <td>TA</td>
      <td>130</td>
      <td>308</td>
      <td>0</td>
      <td>Normal</td>
      <td>98</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>416</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>260</td>
      <td>0</td>
      <td>ST</td>
      <td>112</td>
      <td>Y</td>
      <td>3.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>359</th>
      <td>53</td>
      <td>M</td>
      <td>NAP</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>115</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>311</th>
      <td>60</td>
      <td>M</td>
      <td>ASY</td>
      <td>125</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>110</td>
      <td>N</td>
      <td>0.1</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>580</th>
      <td>51</td>
      <td>M</td>
      <td>ASY</td>
      <td>131</td>
      <td>152</td>
      <td>1</td>
      <td>LVH</td>
      <td>130</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>707</th>
      <td>54</td>
      <td>M</td>
      <td>ASY</td>
      <td>124</td>
      <td>266</td>
      <td>0</td>
      <td>LVH</td>
      <td>109</td>
      <td>Y</td>
      <td>2.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>757</th>
      <td>50</td>
      <td>M</td>
      <td>NAP</td>
      <td>140</td>
      <td>233</td>
      <td>0</td>
      <td>Normal</td>
      <td>163</td>
      <td>N</td>
      <td>0.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>891</th>
      <td>66</td>
      <td>F</td>
      <td>NAP</td>
      <td>146</td>
      <td>278</td>
      <td>0</td>
      <td>LVH</td>
      <td>152</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>396</th>
      <td>62</td>
      <td>F</td>
      <td>TA</td>
      <td>140</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>143</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>529</th>
      <td>72</td>
      <td>M</td>
      <td>ASY</td>
      <td>143</td>
      <td>211</td>
      <td>0</td>
      <td>Normal</td>
      <td>109</td>
      <td>Y</td>
      <td>1.4</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>518</th>
      <td>48</td>
      <td>M</td>
      <td>NAP</td>
      <td>102</td>
      <td>0</td>
      <td>1</td>
      <td>ST</td>
      <td>110</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>351</th>
      <td>43</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>140</td>
      <td>Y</td>
      <td>0.5</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>275</th>
      <td>59</td>
      <td>M</td>
      <td>NAP</td>
      <td>180</td>
      <td>213</td>
      <td>0</td>
      <td>Normal</td>
      <td>100</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>158</th>
      <td>44</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>290</td>
      <td>0</td>
      <td>Normal</td>
      <td>100</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>528</th>
      <td>49</td>
      <td>M</td>
      <td>NAP</td>
      <td>131</td>
      <td>142</td>
      <td>0</td>
      <td>Normal</td>
      <td>127</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>854</th>
      <td>52</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>325</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.2</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>575</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>137</td>
      <td>282</td>
      <td>1</td>
      <td>Normal</td>
      <td>126</td>
      <td>Y</td>
      <td>1.2</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>342</th>
      <td>61</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>113</td>
      <td>N</td>
      <td>1.4</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>766</th>
      <td>50</td>
      <td>F</td>
      <td>NAP</td>
      <td>120</td>
      <td>219</td>
      <td>0</td>
      <td>Normal</td>
      <td>158</td>
      <td>N</td>
      <td>1.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>855</th>
      <td>68</td>
      <td>M</td>
      <td>NAP</td>
      <td>180</td>
      <td>274</td>
      <td>1</td>
      <td>LVH</td>
      <td>150</td>
      <td>Y</td>
      <td>1.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>456</th>
      <td>61</td>
      <td>M</td>
      <td>NAP</td>
      <td>120</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>80</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>533</th>
      <td>63</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>252</td>
      <td>0</td>
      <td>ST</td>
      <td>140</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>615</th>
      <td>70</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>322</td>
      <td>0</td>
      <td>LVH</td>
      <td>109</td>
      <td>N</td>
      <td>2.4</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>367</th>
      <td>68</td>
      <td>M</td>
      <td>ASY</td>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>ST</td>
      <td>120</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>370</th>
      <td>60</td>
      <td>M</td>
      <td>ASY</td>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>63</td>
      <td>Y</td>
      <td>0.5</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>486</th>
      <td>55</td>
      <td>M</td>
      <td>ATA</td>
      <td>110</td>
      <td>214</td>
      <td>1</td>
      <td>ST</td>
      <td>180</td>
      <td>N</td>
      <td>0.4</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>665</th>
      <td>42</td>
      <td>M</td>
      <td>ASY</td>
      <td>136</td>
      <td>315</td>
      <td>0</td>
      <td>Normal</td>
      <td>125</td>
      <td>Y</td>
      <td>1.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>669</th>
      <td>45</td>
      <td>F</td>
      <td>ATA</td>
      <td>130</td>
      <td>234</td>
      <td>0</td>
      <td>LVH</td>
      <td>175</td>
      <td>N</td>
      <td>0.6</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>212</th>
      <td>56</td>
      <td>M</td>
      <td>NAP</td>
      <td>130</td>
      <td>276</td>
      <td>0</td>
      <td>Normal</td>
      <td>128</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>79</th>
      <td>49</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>206</td>
      <td>0</td>
      <td>Normal</td>
      <td>170</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>148</th>
      <td>50</td>
      <td>M</td>
      <td>ATA</td>
      <td>120</td>
      <td>168</td>
      <td>0</td>
      <td>Normal</td>
      <td>160</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>500</th>
      <td>65</td>
      <td>M</td>
      <td>ASY</td>
      <td>136</td>
      <td>248</td>
      <td>0</td>
      <td>Normal</td>
      <td>140</td>
      <td>Y</td>
      <td>4.0</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>710</th>
      <td>47</td>
      <td>M</td>
      <td>ASY</td>
      <td>110</td>
      <td>275</td>
      <td>0</td>
      <td>LVH</td>
      <td>118</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>631</th>
      <td>46</td>
      <td>M</td>
      <td>ASY</td>
      <td>140</td>
      <td>311</td>
      <td>0</td>
      <td>Normal</td>
      <td>120</td>
      <td>Y</td>
      <td>1.8</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>394</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>0</td>
      <td>1</td>
      <td>Normal</td>
      <td>98</td>
      <td>Y</td>
      <td>2.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>594</th>
      <td>58</td>
      <td>M</td>
      <td>ASY</td>
      <td>160</td>
      <td>256</td>
      <td>1</td>
      <td>LVH</td>
      <td>113</td>
      <td>Y</td>
      <td>1.0</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>133</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>150</td>
      <td>230</td>
      <td>0</td>
      <td>ST</td>
      <td>124</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>813</th>
      <td>69</td>
      <td>F</td>
      <td>TA</td>
      <td>140</td>
      <td>239</td>
      <td>0</td>
      <td>Normal</td>
      <td>151</td>
      <td>N</td>
      <td>1.8</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>734</th>
      <td>56</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>283</td>
      <td>1</td>
      <td>LVH</td>
      <td>103</td>
      <td>Y</td>
      <td>1.6</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>360</th>
      <td>62</td>
      <td>M</td>
      <td>NAP</td>
      <td>160</td>
      <td>0</td>
      <td>0</td>
      <td>Normal</td>
      <td>72</td>
      <td>Y</td>
      <td>0.0</td>
      <td>Flat</td>
    </tr>
    <tr>
      <th>875</th>
      <td>58</td>
      <td>F</td>
      <td>NAP</td>
      <td>120</td>
      <td>340</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>108</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- We have lift from 0.8804 to .9094


```python
#trial = np.array([[55,'M','TA',120,0,0,'Normal',122,'N',1.0,'Flat']])
```


```python
#trial
```




    array([['55', 'M', 'TA', '120', '0', '0', 'Normal', '122', 'N', '1.0',
            'Flat']], dtype='<U32')



<a id="14"></a>
<font color="lightseagreen" size=+1.5><b>Feature Importance</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
#res = model.predict(trial)
```


```python
#res
```




    array([1], dtype=int64)




```python
# import pickle
# data = {"model": model}
```


```python
# with open('saved_steps.pkl', 'wb') as file:
#     pickle.dump(data, file)
```


```python
# with open('saved_steps.pkl', 'rb') as file:
#     data = pickle.load(file)
```


```python
# model_loded = data['model']
```


```python
# new_res = model_loded.predict(trial)
```


```python
# new_res
```




    array([1], dtype=int64)




```python
feature_importance = np.array(model.get_feature_importance())
features = np.array(X_train.columns)
fi={'features':features,'feature_importance':feature_importance}
df_fi = pd.DataFrame(fi)
df_fi.sort_values(by=['feature_importance'], ascending=True,inplace=True)
fig = px.bar(df_fi, x='feature_importance', y='features',title="CatBoost Feature Importance",height=500)
fig.show()
```


<div>                            <div id="c7439585-c222-4343-a823-452ee4b49a97" class="plotly-graph-div" style="height:500px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c7439585-c222-4343-a823-452ee4b49a97")) {                    Plotly.newPlot(                        "c7439585-c222-4343-a823-452ee4b49a97",                        [{"alignmentgroup":"True","hovertemplate":"feature_importance=%{x}<br>features=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"h","showlegend":false,"textposition":"auto","type":"bar","x":[5.855926793112449,5.93342908650591,6.7475431029453725,7.095111890238894,7.604893046891047,7.71991886753339,7.756085511256721,9.448012106843612,10.301741574595345,12.004999274618992,19.53233874545823],"xaxis":"x","y":["MaxHR","FastingBS","Sex","RestingBP","Age","Cholesterol","RestingECG","Oldpeak","ExerciseAngina","ChestPainType","ST_Slope"],"yaxis":"y"}],                        {"barmode":"relative","height":500,"legend":{"tracegroupgap":0},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"CatBoost Feature Importance"},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"feature_importance"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"features"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c7439585-c222-4343-a823-452ee4b49a97');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id="15"></a>
<font color="lightseagreen" size=+1.5><b>Model Comparison</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


```python
result_final = pd.concat([dummy_result_df,result_df1,result_df2,result_df3,result_df4,result_df5,result_df6],axis=0)
```


```python
result_final.sort_values(by=['Accuracy'], ascending=True,inplace=True)
fig = px.bar(result_final, x='Accuracy', y=result_final.index,title='Model Comparison',height=600,labels={'index':'MODELS'})
fig.show()
```


<div>                            <div id="7126b4b6-8508-4d5f-8e30-c450fd88669c" class="plotly-graph-div" style="height:600px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7126b4b6-8508-4d5f-8e30-c450fd88669c")) {                    Plotly.newPlot(                        "7126b4b6-8508-4d5f-8e30-c450fd88669c",                        [{"alignmentgroup":"True","hovertemplate":"Accuracy=%{x}<br>MODELS=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"h","showlegend":false,"textposition":"auto","type":"bar","x":[0.5942,0.7174,0.7246,0.8297,0.8659,0.8696,0.8696,0.8732,0.8768,0.8804,0.8804,0.8804,0.8841,0.8841,0.8841,0.8877,0.9094],"xaxis":"x","y":["DummyClassifier","KNeighbors","SVM","XGBoost","Ada","LinearDiscriminant","LinearDiscriminant_scl","LightGBM","Gradient","Logistic_scl","Catboost_default","ExtraTree","KNeighbors_scl","SVM_scl","Logistic","Random","Catboost_tuned"],"yaxis":"y"}],                        {"barmode":"relative","height":600,"legend":{"tracegroupgap":0},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Model Comparison"},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Accuracy"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"MODELS"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('7126b4b6-8508-4d5f-8e30-c450fd88669c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


<a id="16"></a>
<font color="darkblue" size=+1.5><b>Conclusion</b></font>

<a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>


- We have developed model to classifiy heart disease cases.

- First, we  made the detailed exploratory analysis.
- We have decided which metric to use.
- We analyzed both target and features in detail.
- We transform categorical variables into numeric so we can use them in the model.
- We use pipeline to avoid data leakage.
- We looked at the results of the each model and selected the best one for the problem on hand.
- We looked in detail Catboost
- We made hyperparameter tuning of the Catboost with Optuna to see the improvement
- We looked at the feature importance.



'''

def shownotebook():
    data = open('notebook.ipynb')
    st.download_button(label='Download Notebook', data=data, file_name ="heart_failure_notebook.ipynb", mime="text/ipynb"  )
    st.markdown("---")
    st.markdown(nb)
    