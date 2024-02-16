# Capstone-Project-Pt.3
```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


```


```python
import os
os.getcwd()
```




    'C:\\Users\\frank_jk22l4o'




```python
file_path = os.path.abspath('creditcard.csv')
print("File Path", file_path)
```

    File Path C:\Users\frank_jk22l4o\creditcard.csv
    


```python
new_directory = "C:\\Users\\frank_jk22l4o\downloads"
os.chdir(new_directory)
```


```python
cc = pd.read_csv('creditcard.csv')
```


```python
cc.head(10)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>




```python
print("\033[91m This is a Very Detailed Fraud Detection Analysis")
cc.info()
```

    [91m This is a Very Detailed Fraud Detection Analysis
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    


```python
cc.describe()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>1.759061e-12</td>
      <td>-8.251130e-13</td>
      <td>-9.654937e-13</td>
      <td>8.321385e-13</td>
      <td>1.649999e-13</td>
      <td>4.248366e-13</td>
      <td>-3.054600e-13</td>
      <td>8.777971e-14</td>
      <td>-1.179749e-12</td>
      <td>...</td>
      <td>-3.405756e-13</td>
      <td>-5.723197e-13</td>
      <td>-9.725856e-13</td>
      <td>1.464150e-12</td>
      <td>-6.987102e-13</td>
      <td>-5.617874e-13</td>
      <td>3.332082e-12</td>
      <td>-3.518874e-12</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
cc.isna().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
cc_cleaned = cc.dropna()
```


```python
print(cc_cleaned)
```

                Time         V1         V2        V3        V4        V5  \
    0            0.0  -1.359807  -0.072781  2.536347  1.378155 -0.338321   
    1            0.0   1.191857   0.266151  0.166480  0.448154  0.060018   
    2            1.0  -1.358354  -1.340163  1.773209  0.379780 -0.503198   
    3            1.0  -0.966272  -0.185226  1.792993 -0.863291 -0.010309   
    4            2.0  -1.158233   0.877737  1.548718  0.403034 -0.407193   
    ...          ...        ...        ...       ...       ...       ...   
    284802  172786.0 -11.881118  10.071785 -9.834783 -2.066656 -5.364473   
    284803  172787.0  -0.732789  -0.055080  2.035030 -0.738589  0.868229   
    284804  172788.0   1.919565  -0.301254 -3.249640 -0.557828  2.630515   
    284805  172788.0  -0.240440   0.530483  0.702510  0.689799 -0.377961   
    284806  172792.0  -0.533413  -0.189733  0.703337 -0.506271 -0.012546   
    
                  V6        V7        V8        V9  ...       V21       V22  \
    0       0.462388  0.239599  0.098698  0.363787  ... -0.018307  0.277838   
    1      -0.082361 -0.078803  0.085102 -0.255425  ... -0.225775 -0.638672   
    2       1.800499  0.791461  0.247676 -1.514654  ...  0.247998  0.771679   
    3       1.247203  0.237609  0.377436 -1.387024  ... -0.108300  0.005274   
    4       0.095921  0.592941 -0.270533  0.817739  ... -0.009431  0.798278   
    ...          ...       ...       ...       ...  ...       ...       ...   
    284802 -2.606837 -4.918215  7.305334  1.914428  ...  0.213454  0.111864   
    284803  1.058415  0.024330  0.294869  0.584800  ...  0.214205  0.924384   
    284804  3.031260 -0.296827  0.708417  0.432454  ...  0.232045  0.578229   
    284805  0.623708 -0.686180  0.679145  0.392087  ...  0.265245  0.800049   
    284806 -0.649617  1.577006 -0.414650  0.486180  ...  0.261057  0.643078   
    
                 V23       V24       V25       V26       V27       V28  Amount  \
    0      -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62   
    1       0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69   
    2       0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66   
    3      -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50   
    4      -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99   
    ...          ...       ...       ...       ...       ...       ...     ...   
    284802  1.014480 -0.509348  1.436807  0.250034  0.943651  0.823731    0.77   
    284803  0.012463 -1.016226 -0.606624 -0.395255  0.068472 -0.053527   24.79   
    284804 -0.037501  0.640134  0.265745 -0.087371  0.004455 -0.026561   67.88   
    284805 -0.163298  0.123205 -0.569159  0.546668  0.108821  0.104533   10.00   
    284806  0.376777  0.008797 -0.473649 -0.818267 -0.002415  0.013649  217.00   
    
            Class  
    0           0  
    1           0  
    2           0  
    3           0  
    4           0  
    ...       ...  
    284802      0  
    284803      0  
    284804      0  
    284805      0  
    284806      0  
    
    [284807 rows x 31 columns]
    


```python
cc_cleaned.head(20)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>1.449044</td>
      <td>-1.176339</td>
      <td>0.913860</td>
      <td>-1.375667</td>
      <td>-1.971383</td>
      <td>-0.629152</td>
      <td>-1.423236</td>
      <td>0.048456</td>
      <td>-1.720408</td>
      <td>...</td>
      <td>-0.009302</td>
      <td>0.313894</td>
      <td>0.027740</td>
      <td>0.500512</td>
      <td>0.251367</td>
      <td>-0.129478</td>
      <td>0.042850</td>
      <td>0.016253</td>
      <td>7.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0</td>
      <td>0.384978</td>
      <td>0.616109</td>
      <td>-0.874300</td>
      <td>-0.094019</td>
      <td>2.924584</td>
      <td>3.317027</td>
      <td>0.470455</td>
      <td>0.538247</td>
      <td>-0.558895</td>
      <td>...</td>
      <td>0.049924</td>
      <td>0.238422</td>
      <td>0.009130</td>
      <td>0.996710</td>
      <td>-0.767315</td>
      <td>-0.492208</td>
      <td>0.042472</td>
      <td>-0.054337</td>
      <td>9.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10.0</td>
      <td>1.249999</td>
      <td>-1.221637</td>
      <td>0.383930</td>
      <td>-1.234899</td>
      <td>-1.485419</td>
      <td>-0.753230</td>
      <td>-0.689405</td>
      <td>-0.227487</td>
      <td>-2.094011</td>
      <td>...</td>
      <td>-0.231809</td>
      <td>-0.483285</td>
      <td>0.084668</td>
      <td>0.392831</td>
      <td>0.161135</td>
      <td>-0.354990</td>
      <td>0.026416</td>
      <td>0.042422</td>
      <td>121.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11.0</td>
      <td>1.069374</td>
      <td>0.287722</td>
      <td>0.828613</td>
      <td>2.712520</td>
      <td>-0.178398</td>
      <td>0.337544</td>
      <td>-0.096717</td>
      <td>0.115982</td>
      <td>-0.221083</td>
      <td>...</td>
      <td>-0.036876</td>
      <td>0.074412</td>
      <td>-0.071407</td>
      <td>0.104744</td>
      <td>0.548265</td>
      <td>0.104094</td>
      <td>0.021491</td>
      <td>0.021293</td>
      <td>27.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.0</td>
      <td>-2.791855</td>
      <td>-0.327771</td>
      <td>1.641750</td>
      <td>1.767473</td>
      <td>-0.136588</td>
      <td>0.807596</td>
      <td>-0.422911</td>
      <td>-1.907107</td>
      <td>0.755713</td>
      <td>...</td>
      <td>1.151663</td>
      <td>0.222182</td>
      <td>1.020586</td>
      <td>0.028317</td>
      <td>-0.232746</td>
      <td>-0.235557</td>
      <td>-0.164778</td>
      <td>-0.030154</td>
      <td>58.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>12.0</td>
      <td>-0.752417</td>
      <td>0.345485</td>
      <td>2.057323</td>
      <td>-1.468643</td>
      <td>-1.158394</td>
      <td>-0.077850</td>
      <td>-0.608581</td>
      <td>0.003603</td>
      <td>-0.436167</td>
      <td>...</td>
      <td>0.499625</td>
      <td>1.353650</td>
      <td>-0.256573</td>
      <td>-0.065084</td>
      <td>-0.039124</td>
      <td>-0.087086</td>
      <td>-0.180998</td>
      <td>0.129394</td>
      <td>15.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12.0</td>
      <td>1.103215</td>
      <td>-0.040296</td>
      <td>1.267332</td>
      <td>1.289091</td>
      <td>-0.735997</td>
      <td>0.288069</td>
      <td>-0.586057</td>
      <td>0.189380</td>
      <td>0.782333</td>
      <td>...</td>
      <td>-0.024612</td>
      <td>0.196002</td>
      <td>0.013802</td>
      <td>0.103758</td>
      <td>0.364298</td>
      <td>-0.382261</td>
      <td>0.092809</td>
      <td>0.037051</td>
      <td>12.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13.0</td>
      <td>-0.436905</td>
      <td>0.918966</td>
      <td>0.924591</td>
      <td>-0.727219</td>
      <td>0.915679</td>
      <td>-0.127867</td>
      <td>0.707642</td>
      <td>0.087962</td>
      <td>-0.665271</td>
      <td>...</td>
      <td>-0.194796</td>
      <td>-0.672638</td>
      <td>-0.156858</td>
      <td>-0.888386</td>
      <td>-0.342413</td>
      <td>-0.049027</td>
      <td>0.079692</td>
      <td>0.131024</td>
      <td>0.89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>14.0</td>
      <td>-5.401258</td>
      <td>-5.450148</td>
      <td>1.186305</td>
      <td>1.736239</td>
      <td>3.049106</td>
      <td>-1.763406</td>
      <td>-1.559738</td>
      <td>0.160842</td>
      <td>1.233090</td>
      <td>...</td>
      <td>-0.503600</td>
      <td>0.984460</td>
      <td>2.458589</td>
      <td>0.042119</td>
      <td>-0.481631</td>
      <td>-0.621272</td>
      <td>0.392053</td>
      <td>0.949594</td>
      <td>46.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>15.0</td>
      <td>1.492936</td>
      <td>-1.029346</td>
      <td>0.454795</td>
      <td>-1.438026</td>
      <td>-1.555434</td>
      <td>-0.720961</td>
      <td>-1.080664</td>
      <td>-0.053127</td>
      <td>-1.978682</td>
      <td>...</td>
      <td>-0.177650</td>
      <td>-0.175074</td>
      <td>0.040002</td>
      <td>0.295814</td>
      <td>0.332931</td>
      <td>-0.220385</td>
      <td>0.022298</td>
      <td>0.007602</td>
      <td>5.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 31 columns</p>
</div>




```python
# Checking For Missing values
missingdata = cc_cleaned.isnull().sum()
missingdata
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
sns.heatmap(cc_cleaned.isnull(), cbar=False, cmap='viridis')
plt.show()
```


    
![png](output_13_0.png)
    



```python
def transaction():
    for i in range(30):
        print("-", end="")
```


```python
transaction()
print("\033[91m\n1 = Fraudulent\t0 = Legit")
transaction()

cc_cleaned.Class.value_counts()
```

    ------------------------------[91m
    1 = Fraudulent	0 = Legit
    ------------------------------




    Class
    0    284315
    1       492
    Name: count, dtype: int64




```python
cc_cleaned.groupby('Class').mean()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>94838.202258</td>
      <td>0.008258</td>
      <td>-0.006271</td>
      <td>0.012171</td>
      <td>-0.007860</td>
      <td>0.005453</td>
      <td>0.002419</td>
      <td>0.009637</td>
      <td>-0.000987</td>
      <td>0.004467</td>
      <td>...</td>
      <td>-0.000644</td>
      <td>-0.001235</td>
      <td>-0.000024</td>
      <td>0.000070</td>
      <td>0.000182</td>
      <td>-0.000072</td>
      <td>-0.000089</td>
      <td>-0.000295</td>
      <td>-0.000131</td>
      <td>88.291022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80746.806911</td>
      <td>-4.771948</td>
      <td>3.623778</td>
      <td>-7.033281</td>
      <td>4.542029</td>
      <td>-3.151225</td>
      <td>-1.397737</td>
      <td>-5.568731</td>
      <td>0.570636</td>
      <td>-2.581123</td>
      <td>...</td>
      <td>0.372319</td>
      <td>0.713588</td>
      <td>0.014049</td>
      <td>-0.040308</td>
      <td>-0.105130</td>
      <td>0.041449</td>
      <td>0.051648</td>
      <td>0.170575</td>
      <td>0.075667</td>
      <td>122.211321</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>




```python
cc_cleaned.groupby('Class')['Amount'].mean()
```




    Class
    0     88.291022
    1    122.211321
    Name: Amount, dtype: float64




```python
# For Above ^ 88.9 is the average mean of data that lies in Class 0 and that are normal transactions.
# For Above ^ 122.2 is the average mean of data that lies in Class 1 and that are faulty transactions.
```


```python
# Normalizing the data

norm_trans = cc_cleaned[cc_cleaned.Class==0]
f_trans = cc_cleaned[cc_cleaned.Class==1]
```


```python
print(norm_trans.shape,f_trans.shape)
```

    (284315, 31) (492, 31)
    


```python
norm_trans = norm_trans.sample(n=492)
```


```python
norm_trans.shape
```




    (492, 31)




```python
cc_new = pd.concat([norm_trans,f_trans])
```


```python
cc_new.head(5)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138551</th>
      <td>82718.0</td>
      <td>-0.144511</td>
      <td>0.224121</td>
      <td>1.109071</td>
      <td>0.846746</td>
      <td>-1.543769</td>
      <td>0.744762</td>
      <td>1.082263</td>
      <td>-0.019542</td>
      <td>0.172396</td>
      <td>...</td>
      <td>0.228865</td>
      <td>0.732812</td>
      <td>0.343975</td>
      <td>0.072260</td>
      <td>-0.586382</td>
      <td>-0.482619</td>
      <td>0.039082</td>
      <td>-0.108141</td>
      <td>300.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51725</th>
      <td>45090.0</td>
      <td>-0.576516</td>
      <td>0.593863</td>
      <td>1.790672</td>
      <td>0.187901</td>
      <td>-0.116811</td>
      <td>-0.515673</td>
      <td>0.943212</td>
      <td>-0.287863</td>
      <td>0.174292</td>
      <td>...</td>
      <td>-0.187207</td>
      <td>-0.310514</td>
      <td>-0.154360</td>
      <td>0.389851</td>
      <td>-0.143983</td>
      <td>0.184925</td>
      <td>-0.210263</td>
      <td>-0.115712</td>
      <td>52.04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>183656</th>
      <td>125896.0</td>
      <td>1.941576</td>
      <td>-0.032807</td>
      <td>-0.610152</td>
      <td>1.646305</td>
      <td>-0.097767</td>
      <td>-0.350460</td>
      <td>0.052045</td>
      <td>-0.166319</td>
      <td>0.934865</td>
      <td>...</td>
      <td>-0.316661</td>
      <td>-0.450065</td>
      <td>0.224125</td>
      <td>-0.012676</td>
      <td>0.083353</td>
      <td>-0.820505</td>
      <td>0.049741</td>
      <td>-0.040756</td>
      <td>10.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>132479</th>
      <td>80003.0</td>
      <td>0.789921</td>
      <td>-1.106316</td>
      <td>0.871156</td>
      <td>0.369516</td>
      <td>-1.302417</td>
      <td>0.128324</td>
      <td>-0.545363</td>
      <td>0.204605</td>
      <td>1.345234</td>
      <td>...</td>
      <td>-0.100489</td>
      <td>-0.491449</td>
      <td>-0.049019</td>
      <td>0.140852</td>
      <td>-0.045425</td>
      <td>0.954680</td>
      <td>-0.061359</td>
      <td>0.042678</td>
      <td>203.96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>275876</th>
      <td>166766.0</td>
      <td>1.713539</td>
      <td>0.211588</td>
      <td>-0.616591</td>
      <td>3.675128</td>
      <td>0.421280</td>
      <td>0.460982</td>
      <td>0.118945</td>
      <td>0.030522</td>
      <td>-1.133704</td>
      <td>...</td>
      <td>0.120524</td>
      <td>0.204315</td>
      <td>0.054054</td>
      <td>-0.480522</td>
      <td>-0.103266</td>
      <td>-0.062018</td>
      <td>-0.037882</td>
      <td>-0.040450</td>
      <td>97.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
cc_new.tail(5)
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>279863</th>
      <td>169142.0</td>
      <td>-1.927883</td>
      <td>1.125653</td>
      <td>-4.518331</td>
      <td>1.749293</td>
      <td>-1.566487</td>
      <td>-2.010494</td>
      <td>-0.882850</td>
      <td>0.697211</td>
      <td>-2.064945</td>
      <td>...</td>
      <td>0.778584</td>
      <td>-0.319189</td>
      <td>0.639419</td>
      <td>-0.294885</td>
      <td>0.537503</td>
      <td>0.788395</td>
      <td>0.292680</td>
      <td>0.147968</td>
      <td>390.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280143</th>
      <td>169347.0</td>
      <td>1.378559</td>
      <td>1.289381</td>
      <td>-5.004247</td>
      <td>1.411850</td>
      <td>0.442581</td>
      <td>-1.326536</td>
      <td>-1.413170</td>
      <td>0.248525</td>
      <td>-1.127396</td>
      <td>...</td>
      <td>0.370612</td>
      <td>0.028234</td>
      <td>-0.145640</td>
      <td>-0.081049</td>
      <td>0.521875</td>
      <td>0.739467</td>
      <td>0.389152</td>
      <td>0.186637</td>
      <td>0.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280149</th>
      <td>169351.0</td>
      <td>-0.676143</td>
      <td>1.126366</td>
      <td>-2.213700</td>
      <td>0.468308</td>
      <td>-1.120541</td>
      <td>-0.003346</td>
      <td>-2.234739</td>
      <td>1.210158</td>
      <td>-0.652250</td>
      <td>...</td>
      <td>0.751826</td>
      <td>0.834108</td>
      <td>0.190944</td>
      <td>0.032070</td>
      <td>-0.739695</td>
      <td>0.471111</td>
      <td>0.385107</td>
      <td>0.194361</td>
      <td>77.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281144</th>
      <td>169966.0</td>
      <td>-3.113832</td>
      <td>0.585864</td>
      <td>-5.399730</td>
      <td>1.817092</td>
      <td>-0.840618</td>
      <td>-2.943548</td>
      <td>-2.208002</td>
      <td>1.058733</td>
      <td>-1.632333</td>
      <td>...</td>
      <td>0.583276</td>
      <td>-0.269209</td>
      <td>-0.456108</td>
      <td>-0.183659</td>
      <td>-0.328168</td>
      <td>0.606116</td>
      <td>0.884876</td>
      <td>-0.253700</td>
      <td>245.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281674</th>
      <td>170348.0</td>
      <td>1.991976</td>
      <td>0.158476</td>
      <td>-2.583441</td>
      <td>0.408670</td>
      <td>1.151147</td>
      <td>-0.096695</td>
      <td>0.223050</td>
      <td>-0.068384</td>
      <td>0.577829</td>
      <td>...</td>
      <td>-0.164350</td>
      <td>-0.295135</td>
      <td>-0.072173</td>
      <td>-0.450261</td>
      <td>0.313267</td>
      <td>-0.289617</td>
      <td>0.002988</td>
      <td>-0.015309</td>
      <td>42.53</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
# Data Analysis & Validation

sns.histplot(data=cc_new,x='Time',color='green',bins=50,kde=True)
plt.title('Detailed Fraud Detection Analysis (View1)')
plt.xlabel('Amount per Occurance')
plt.ylabel('Number of Occurances')
plt.show
```




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_26_1.png)
    



```python
plt.figure(figsize=(4,3))
sns.histplot(data=cc_new, x='Amount',bins=35, color='blue')
```




    <Axes: xlabel='Amount', ylabel='Count'>




    
![png](output_27_1.png)
    



```python
plt.figure(figsize=(2,3))
sns.countplot(data=cc_new,x='Class',width= 1.0)
```




    <Axes: xlabel='Class', ylabel='count'>




    
![png](output_28_1.png)
    



```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
```


```python
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=42) 
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
Pipeline([ ('oversample', SMOTE(random_state=42)), ('undersample', RandomUnderSampler(random_state=42)), ('model', RandomForestClassifier(random_state=42)) ]) 
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;oversample&#x27;, SMOTE(random_state=42)),
                (&#x27;undersample&#x27;, RandomUnderSampler(random_state=42)),
                (&#x27;model&#x27;, RandomForestClassifier(random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;Pipeline<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></label><div class="sk-toggleable__content "><pre>Pipeline(steps=[(&#x27;oversample&#x27;, SMOTE(random_state=42)),
                (&#x27;undersample&#x27;, RandomUnderSampler(random_state=42)),
                (&#x27;model&#x27;, RandomForestClassifier(random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow ">SMOTE</label><div class="sk-toggleable__content "><pre>SMOTE(random_state=42)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow ">RandomUnderSampler</label><div class="sk-toggleable__content "><pre>RandomUnderSampler(random_state=42)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></label><div class="sk-toggleable__content "><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div></div></div>




```python
pipeline = SMOTE(random_state=42)
```


```python
X_train, y_train = pipeline.fit_resample(X_train, y_train) 
```


```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification 
```


```python
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=42) 
```


```python
smote = SMOTE(random_state=42) 
```


```python
X_resampled, y_resampled = smote.fit_resample(X, y) 

```


```python
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
from sklearn.datasets import load_iris
```


```python
iris = load_iris()
X, y = iris.data, iris.target 
```


```python
model = SVC() 
```


```python
param_grid = {'C': [0.1, 1, 10, 100]} 
```


```python
grid_search = GridSearchCV(model, param_grid, cv=5) 
```


```python
grid_search.fit(X, y) 
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=SVC(), param_grid={&#x27;C&#x27;: [0.1, 1, 10, 100]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GridSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5, estimator=SVC(), param_grid={&#x27;C&#x27;: [0.1, 1, 10, 100]})</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: SVC</label><div class="sk-toggleable__content fitted"><pre>SVC()</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a></label><div class="sk-toggleable__content fitted"><pre>SVC()</pre></div> </div></div></div></div></div></div></div></div></div>




```python
print("Best Parameters: ", grid_search.best_params_) 
print("Best Accuracy: ", grid_search.best_score_)
```

    Best Parameters:  {'C': 10}
    Best Accuracy:  0.9800000000000001
    


```python
# Train- Test Split, Hyper- Parameter Tuning & Model Selection
```


```python
x = cc_new.drop(['Class'],axis='columns').values
y = cc_new.Class
```


```python
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.5)
```


```python
print(x_train.shape,x_test.shape)
```

    (492, 30) (492, 30)
    


```python
model_params = {
    
    'logistic_reg':{
            'model':LogisticRegression(),
            'params':{
                'C':[1,5,10,15,20]
            }
    },
    'randomforest':{
            'model':RandomForestClassifier(),
            'params':{
                'n_estimators':[5,10,15,20,25]
            }
    }
}
```


```python
score = []
for mod_name,mod in model_params.items():
    rscv = GridSearchCV(mod['model'],mod['params'],cv=2,return_train_score=False)
    rscv.fit(x_train,y_train)
    score.append({'model':mod_name,'best_score':rscv.best_score_,'best_parms':rscv.best_params_})
```


```python
score
```




    [{'model': 'logistic_reg',
      'best_score': 0.9268292682926829,
      'best_parms': {'C': 20}},
     {'model': 'randomforest',
      'best_score': 0.9349593495934959,
      'best_parms': {'n_estimators': 20}}]




```python
model = LogisticRegression(C=10).fit(x_train,y_train)
```


```python
model.score(x_test,y_test)
```




    0.9308943089430894




```python
y_predition = model.predict(x_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(classification_report(y_test,y_predition))
```

                  precision    recall  f1-score   support
    
               0       0.90      0.97      0.93       246
               1       0.97      0.89      0.93       246
    
        accuracy                           0.93       492
       macro avg       0.93      0.93      0.93       492
    weighted avg       0.93      0.93      0.93       492
    
    


```python
matrix = confusion_matrix(y_test,y_predition)
```


```python
matrix
```




    array([[239,   7],
           [ 27, 219]], dtype=int64)




```python
plt.figure(figsize=(4,3))
sns.heatmap(matrix,annot=True,fmt='.2f',cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actuals')
plt.show()
```


    
![png](output_60_0.png)
    



```python
# Predictions
```


```python
input_data = x_test[12]
```


```python
input_data
```




    array([ 4.26940000e+04,  1.13072349e+00,  2.66213080e-02,  2.52567793e-01,
            1.21867067e+00, -5.87161290e-02,  3.02518570e-01, -1.26235396e-01,
            2.29569697e-01,  2.61611452e-01,  8.19265830e-02,  6.33420122e-01,
            3.77073641e-01, -1.53540125e+00,  5.18517229e-01, -6.18296255e-01,
           -3.65351273e-01, -2.76075110e-02, -2.05129394e-01,  5.27412500e-02,
           -2.42844929e-01, -8.34006170e-02, -1.20932589e-01, -1.04514983e-01,
           -3.24819736e-01,  6.27793687e-01, -3.06013085e-01,  2.53206750e-02,
            7.67239000e-04,  1.52600000e+01])




```python
model.fit(x_train,y_train)
```




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=10)</pre></div> </div></div></div></div>




```python
pred = model.predict([input_data])
if pred[0] == 1:
    print("It is Fault Transaction!")
else: 
    print("Transaction is good!")
```

    Transaction is good!
    


```python
# Cross Check
```


```python
y_test[12:13]
```




    46236    0
    Name: Class, dtype: int64
