# Refreshment Course for Data Science and Business Intelligence Certification




## Introduction

When we are asked to prove our credibility in a field, for example, as doctors we are required to show our licenses. In many professions, certification is one of the main ways to demonstrate credibility.

For instance, there are internationally recognized certifications such as AWS or GCP. Many countries or third-party institutions also provide their own certifications. In Indonesia, an official government body is responsible for this. In this post, we will discuss some of their certifications in data science and business intelligence, as well as the refresher materials available for preparation.

## Data Science Certification

Each certification comes with predefined units that the test taker must complete in order to become certified. For example, in Data Science, the required units include:
* Data Collection
* Data Exploration
* Data Validation
* Data Definition
* Data Construction
* Data Labeling
* Model Building
* Model Evaluation

While I am not entirely sure whether these units fully reflect the best practices of a modern data science workflow, I believe they could be simplified and better aligned with current standards, such as: 
* Data Collection
* Data Understanding
* Data Exploration
* Data Preprocessing
* Model Development
* Model Deployment

It is also worth noting that these units were most likely designed before the pre-LLM era. Today, with the advancement of generative AI, many of these steps could potentially be automated or significantly accelerated. :D

### Data Collection

The first step is data collection. When taking a certification, the provider usually provides both the dataset and the goal for the test takers. However, there may be cases where you need to find and collect the data yourself. If so, what should you do?

First, you need to determine what kind of data you want to analyze. For example:
* Stock data, if you are role-playing as an investment analyst.
* Revenue data, if you are role-playing as a company analyst aiming to boost revenue.
* Customer churn data, if your goal is to prevent the company from losing customers, etc.

It is important to remember that every type of data comes with its own characteristics, which will influence how you approach analysis.

Second, after deciding on the type of data, you should identify platforms to find open datasets, such as Kaggle. Alternatively, if you have access to proprietary data and permission to use it, you can work with that as long as privacy and confidentiality are preserved.

For simplicity, we will use the Telco customer churn dataset, which can be found [[1]](https://kaggle.com/datasets/blastchar/telco-customer-churn).

### Data Understanding

After deciding on the dataset, it is essential to understand what the data is about. I usually find that reading the data’s description and metadata is very helpful. Here is what the provider states about the Telco dataset:

>Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs. The raw data contains 7043 rows (customers) and 21 columns (features)." [IBM Sample Data Sets]

From this, we know that the dataset contains 7,043 customers with 21 properties. Taking a quick glance at the data, we can notice some properties that are likely of most interest, Churn. By this time, we may come up with several questions, such as:
* What proportion of customers have churned?
* What are the main reasons customers decide to churn?
* How does customer churn affect the company?
* How can we prevent more customers from churning?
* What strategies should we implement?

From my perspective, data science is essentially a workflow designed to answer these questions. This can be achieved by building predictive models to simulate customer behavior and ultimately create the desired outcome; in this case, retaining more customers.


## Data Exploration

After understanding the data and defining the objectives, the next step is usually to conduct Exploratory Data Analysis (EDA) to gain deeper insights. In the conventional approach, this often begins with tasks such as:
* reading the dataset with pandas, 
* performing sampling, 
* generating summaries, 
* and creating basic plots, etc. 

While effective, these steps can be somewhat repetitive, and, if I may say, a little tedious.

#### The Conventional Way (Tedious and Repetitive)




<details>
<summary>show / hide</summary>

```
python

import pandas as pd
df = pd.read_csv(f'{base_path}/telco.csv')
df.sample(10)
```

</details>





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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>285</th>
      <td>6202-DYYFX</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>22</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>76.00</td>
      <td>1783.6</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4323</th>
      <td>3707-LRWZD</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>32</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Electronic check</td>
      <td>84.05</td>
      <td>2781.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5302</th>
      <td>9700-ISPUP</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>10</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>65.50</td>
      <td>616.9</td>
      <td>No</td>
    </tr>
    <tr>
      <th>171</th>
      <td>1875-QIVME</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.40</td>
      <td>242.8</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6739</th>
      <td>6994-KERXL</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>55.90</td>
      <td>238.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>868</th>
      <td>3313-QKNKB</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>59</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Electronic check</td>
      <td>85.55</td>
      <td>5084.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5102</th>
      <td>9070-BCKQP</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>72</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>110.15</td>
      <td>7881.2</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6444</th>
      <td>6302-JGYRJ</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>31</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>79.45</td>
      <td>2587.7</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>4567-AKPIA</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>41</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>109.10</td>
      <td>4454.25</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6360</th>
      <td>8073-IJDCM</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>...</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>20.30</td>
      <td>20.3</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>



#### Automate it! (Minimal Effort, Maximal Gain)




<details>
<summary>show / hide</summary>

```
python

from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile.to_notebook_iframe()
```

</details>








![Auto Eda](https://raw.githubusercontent.com/rezahabibi96/GitBook/refs/heads/main/articles/.resources/refreshment-course-for-data-science-and-business-intelligence-certification/profiling_report.jpeg)



### Data Preprocessing




<details>
<summary>show / hide</summary>

```
python

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder

ids_cols = ['customerID']
num_cols = ['MonthlyCharges', 'tenure']
txt_cols = ['TotalCharges']
tgt_cols = ['Churn']
cat_cols = list(set(df.columns) - set(ids_cols) - set(num_cols) - set(txt_cols) - set(tgt_cols))

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer.set_output(transform="pandas"), num_cols),
    ('cat', categorical_transformer.set_output(transform="pandas"), cat_cols)
])

X = preprocessor.fit_transform(df.drop(columns=['Churn'], axis=1))
y = LabelEncoder().fit_transform(df['Churn'])
```

</details>


### Model Building




<details>
<summary>show / hide</summary>

```
python

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42)
LR = LogisticRegression()

model = LR.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

</details>


                  precision    recall  f1-score   support
    
               0       0.85      0.90      0.88      1282
               1       0.69      0.57      0.63       479
    
        accuracy                           0.81      1761
       macro avg       0.77      0.74      0.75      1761
    weighted avg       0.81      0.81      0.81      1761
    
    

### Low Code Machine Learning with PyCaret

As we can see in the Data Preprocessing and Model Building steps, these often involve repetitive workflows. At this point, we should ask ourselves: Is it really necessary to perform every step manually? Do these steps directly help us answer our main objective, in this case, preventing more customer churn, or could we use more automated tools as long as they allow us to achieve the same goal effectively?




<details>
<summary>show / hide</summary>

```
python

from pycaret.classification import *
exp = ClassificationExperiment()
exp.setup(df.drop(columns=['customerID'], axis=1), target = 'Churn', session_id = 123)
best = exp.compare_models()
```

</details>







<style type="text/css">
#T_6b964 th {
  text-align: left;
}
#T_6b964_row0_col0, #T_6b964_row0_col2, #T_6b964_row1_col0, #T_6b964_row1_col1, #T_6b964_row1_col2, #T_6b964_row1_col3, #T_6b964_row1_col4, #T_6b964_row1_col5, #T_6b964_row1_col6, #T_6b964_row1_col7, #T_6b964_row2_col0, #T_6b964_row2_col1, #T_6b964_row2_col2, #T_6b964_row2_col3, #T_6b964_row2_col4, #T_6b964_row2_col5, #T_6b964_row2_col6, #T_6b964_row2_col7, #T_6b964_row3_col0, #T_6b964_row3_col1, #T_6b964_row3_col3, #T_6b964_row3_col4, #T_6b964_row3_col5, #T_6b964_row3_col6, #T_6b964_row3_col7, #T_6b964_row4_col0, #T_6b964_row4_col1, #T_6b964_row4_col2, #T_6b964_row4_col3, #T_6b964_row4_col4, #T_6b964_row4_col5, #T_6b964_row4_col6, #T_6b964_row4_col7, #T_6b964_row5_col0, #T_6b964_row5_col1, #T_6b964_row5_col2, #T_6b964_row5_col3, #T_6b964_row5_col4, #T_6b964_row5_col5, #T_6b964_row5_col6, #T_6b964_row5_col7, #T_6b964_row6_col0, #T_6b964_row6_col1, #T_6b964_row6_col2, #T_6b964_row6_col3, #T_6b964_row6_col4, #T_6b964_row6_col5, #T_6b964_row6_col6, #T_6b964_row6_col7, #T_6b964_row7_col0, #T_6b964_row7_col1, #T_6b964_row7_col2, #T_6b964_row7_col3, #T_6b964_row7_col4, #T_6b964_row7_col5, #T_6b964_row7_col6, #T_6b964_row7_col7, #T_6b964_row8_col0, #T_6b964_row8_col1, #T_6b964_row8_col2, #T_6b964_row8_col3, #T_6b964_row8_col4, #T_6b964_row8_col5, #T_6b964_row8_col6, #T_6b964_row8_col7, #T_6b964_row9_col0, #T_6b964_row9_col1, #T_6b964_row9_col2, #T_6b964_row9_col3, #T_6b964_row9_col4, #T_6b964_row9_col5, #T_6b964_row9_col6, #T_6b964_row9_col7, #T_6b964_row10_col0, #T_6b964_row10_col1, #T_6b964_row10_col2, #T_6b964_row10_col3, #T_6b964_row10_col4, #T_6b964_row10_col5, #T_6b964_row10_col6, #T_6b964_row10_col7, #T_6b964_row11_col0, #T_6b964_row11_col1, #T_6b964_row11_col2, #T_6b964_row11_col3, #T_6b964_row11_col4, #T_6b964_row11_col5, #T_6b964_row11_col6, #T_6b964_row11_col7, #T_6b964_row12_col0, #T_6b964_row12_col1, #T_6b964_row12_col2, #T_6b964_row12_col3, #T_6b964_row12_col4, #T_6b964_row12_col5, #T_6b964_row12_col6, #T_6b964_row12_col7, #T_6b964_row13_col0, #T_6b964_row13_col1, #T_6b964_row13_col2, #T_6b964_row13_col3, #T_6b964_row13_col4, #T_6b964_row13_col5, #T_6b964_row13_col6, #T_6b964_row13_col7 {
  text-align: left;
}
#T_6b964_row0_col1, #T_6b964_row0_col3, #T_6b964_row0_col4, #T_6b964_row0_col5, #T_6b964_row0_col6, #T_6b964_row0_col7, #T_6b964_row3_col2 {
  text-align: left;
  background-color: yellow;
}
#T_6b964_row0_col8, #T_6b964_row1_col8, #T_6b964_row2_col8, #T_6b964_row3_col8, #T_6b964_row4_col8, #T_6b964_row5_col8, #T_6b964_row6_col8, #T_6b964_row8_col8, #T_6b964_row9_col8, #T_6b964_row10_col8, #T_6b964_row11_col8, #T_6b964_row12_col8, #T_6b964_row13_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_6b964_row7_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_6b964">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6b964_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_6b964_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_6b964_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_6b964_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_6b964_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_6b964_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_6b964_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_6b964_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_6b964_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6b964_level0_row0" class="row_heading level0 row0" >nb</th>
      <td id="T_6b964_row0_col0" class="data row0 col0" >Naive Bayes</td>
      <td id="T_6b964_row0_col1" class="data row0 col1" >0.7789</td>
      <td id="T_6b964_row0_col2" class="data row0 col2" >0.8060</td>
      <td id="T_6b964_row0_col3" class="data row0 col3" >0.7789</td>
      <td id="T_6b964_row0_col4" class="data row0 col4" >0.7727</td>
      <td id="T_6b964_row0_col5" class="data row0 col5" >0.7741</td>
      <td id="T_6b964_row0_col6" class="data row0 col6" >0.4110</td>
      <td id="T_6b964_row0_col7" class="data row0 col7" >0.4139</td>
      <td id="T_6b964_row0_col8" class="data row0 col8" >0.0830</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row1" class="row_heading level0 row1" >knn</th>
      <td id="T_6b964_row1_col0" class="data row1 col0" >K Neighbors Classifier</td>
      <td id="T_6b964_row1_col1" class="data row1 col1" >0.7716</td>
      <td id="T_6b964_row1_col2" class="data row1 col2" >0.7792</td>
      <td id="T_6b964_row1_col3" class="data row1 col3" >0.7716</td>
      <td id="T_6b964_row1_col4" class="data row1 col4" >0.7642</td>
      <td id="T_6b964_row1_col5" class="data row1 col5" >0.7667</td>
      <td id="T_6b964_row1_col6" class="data row1 col6" >0.3912</td>
      <td id="T_6b964_row1_col7" class="data row1 col7" >0.3932</td>
      <td id="T_6b964_row1_col8" class="data row1 col8" >0.0880</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row2" class="row_heading level0 row2" >qda</th>
      <td id="T_6b964_row2_col0" class="data row2 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_6b964_row2_col1" class="data row2 col1" >0.7576</td>
      <td id="T_6b964_row2_col2" class="data row2 col2" >0.7972</td>
      <td id="T_6b964_row2_col3" class="data row2 col3" >0.7576</td>
      <td id="T_6b964_row2_col4" class="data row2 col4" >0.7248</td>
      <td id="T_6b964_row2_col5" class="data row2 col5" >0.7100</td>
      <td id="T_6b964_row2_col6" class="data row2 col6" >0.2137</td>
      <td id="T_6b964_row2_col7" class="data row2 col7" >0.2494</td>
      <td id="T_6b964_row2_col8" class="data row2 col8" >0.0830</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row3" class="row_heading level0 row3" >lr</th>
      <td id="T_6b964_row3_col0" class="data row3 col0" >Logistic Regression</td>
      <td id="T_6b964_row3_col1" class="data row3 col1" >0.7550</td>
      <td id="T_6b964_row3_col2" class="data row3 col2" >0.8296</td>
      <td id="T_6b964_row3_col3" class="data row3 col3" >0.7550</td>
      <td id="T_6b964_row3_col4" class="data row3 col4" >0.7408</td>
      <td id="T_6b964_row3_col5" class="data row3 col5" >0.6890</td>
      <td id="T_6b964_row3_col6" class="data row3 col6" >0.1566</td>
      <td id="T_6b964_row3_col7" class="data row3 col7" >0.2285</td>
      <td id="T_6b964_row3_col8" class="data row3 col8" >0.1650</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row4" class="row_heading level0 row4" >rf</th>
      <td id="T_6b964_row4_col0" class="data row4 col0" >Random Forest Classifier</td>
      <td id="T_6b964_row4_col1" class="data row4 col1" >0.7501</td>
      <td id="T_6b964_row4_col2" class="data row4 col2" >0.7778</td>
      <td id="T_6b964_row4_col3" class="data row4 col3" >0.7501</td>
      <td id="T_6b964_row4_col4" class="data row4 col4" >0.7130</td>
      <td id="T_6b964_row4_col5" class="data row4 col5" >0.6983</td>
      <td id="T_6b964_row4_col6" class="data row4 col6" >0.1819</td>
      <td id="T_6b964_row4_col7" class="data row4 col7" >0.2174</td>
      <td id="T_6b964_row4_col8" class="data row4 col8" >0.1780</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row5" class="row_heading level0 row5" >svm</th>
      <td id="T_6b964_row5_col0" class="data row5 col0" >SVM - Linear Kernel</td>
      <td id="T_6b964_row5_col1" class="data row5 col1" >0.7462</td>
      <td id="T_6b964_row5_col2" class="data row5 col2" >0.7937</td>
      <td id="T_6b964_row5_col3" class="data row5 col3" >0.7462</td>
      <td id="T_6b964_row5_col4" class="data row5 col4" >0.7132</td>
      <td id="T_6b964_row5_col5" class="data row5 col5" >0.6917</td>
      <td id="T_6b964_row5_col6" class="data row5 col6" >0.2081</td>
      <td id="T_6b964_row5_col7" class="data row5 col7" >0.2471</td>
      <td id="T_6b964_row5_col8" class="data row5 col8" >0.0870</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row6" class="row_heading level0 row6" >ridge</th>
      <td id="T_6b964_row6_col0" class="data row6 col0" >Ridge Classifier</td>
      <td id="T_6b964_row6_col1" class="data row6 col1" >0.7385</td>
      <td id="T_6b964_row6_col2" class="data row6 col2" >0.7839</td>
      <td id="T_6b964_row6_col3" class="data row6 col3" >0.7385</td>
      <td id="T_6b964_row6_col4" class="data row6 col4" >0.6907</td>
      <td id="T_6b964_row6_col5" class="data row6 col5" >0.6592</td>
      <td id="T_6b964_row6_col6" class="data row6 col6" >0.0793</td>
      <td id="T_6b964_row6_col7" class="data row6 col7" >0.1278</td>
      <td id="T_6b964_row6_col8" class="data row6 col8" >0.0830</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row7" class="row_heading level0 row7" >lda</th>
      <td id="T_6b964_row7_col0" class="data row7 col0" >Linear Discriminant Analysis</td>
      <td id="T_6b964_row7_col1" class="data row7 col1" >0.7385</td>
      <td id="T_6b964_row7_col2" class="data row7 col2" >0.7338</td>
      <td id="T_6b964_row7_col3" class="data row7 col3" >0.7385</td>
      <td id="T_6b964_row7_col4" class="data row7 col4" >0.6907</td>
      <td id="T_6b964_row7_col5" class="data row7 col5" >0.6592</td>
      <td id="T_6b964_row7_col6" class="data row7 col6" >0.0793</td>
      <td id="T_6b964_row7_col7" class="data row7 col7" >0.1278</td>
      <td id="T_6b964_row7_col8" class="data row7 col8" >0.0780</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row8" class="row_heading level0 row8" >et</th>
      <td id="T_6b964_row8_col0" class="data row8 col0" >Extra Trees Classifier</td>
      <td id="T_6b964_row8_col1" class="data row8 col1" >0.7379</td>
      <td id="T_6b964_row8_col2" class="data row8 col2" >0.7473</td>
      <td id="T_6b964_row8_col3" class="data row8 col3" >0.7379</td>
      <td id="T_6b964_row8_col4" class="data row8 col4" >0.6890</td>
      <td id="T_6b964_row8_col5" class="data row8 col5" >0.6710</td>
      <td id="T_6b964_row8_col6" class="data row8 col6" >0.1069</td>
      <td id="T_6b964_row8_col7" class="data row8 col7" >0.1467</td>
      <td id="T_6b964_row8_col8" class="data row8 col8" >0.1600</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row9" class="row_heading level0 row9" >lightgbm</th>
      <td id="T_6b964_row9_col0" class="data row9 col0" >Light Gradient Boosting Machine</td>
      <td id="T_6b964_row9_col1" class="data row9 col1" >0.7349</td>
      <td id="T_6b964_row9_col2" class="data row9 col2" >0.7311</td>
      <td id="T_6b964_row9_col3" class="data row9 col3" >0.7349</td>
      <td id="T_6b964_row9_col4" class="data row9 col4" >0.7270</td>
      <td id="T_6b964_row9_col5" class="data row9 col5" >0.7271</td>
      <td id="T_6b964_row9_col6" class="data row9 col6" >0.2888</td>
      <td id="T_6b964_row9_col7" class="data row9 col7" >0.2942</td>
      <td id="T_6b964_row9_col8" class="data row9 col8" >0.2100</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row10" class="row_heading level0 row10" >dummy</th>
      <td id="T_6b964_row10_col0" class="data row10 col0" >Dummy Classifier</td>
      <td id="T_6b964_row10_col1" class="data row10 col1" >0.7347</td>
      <td id="T_6b964_row10_col2" class="data row10 col2" >0.5000</td>
      <td id="T_6b964_row10_col3" class="data row10 col3" >0.7347</td>
      <td id="T_6b964_row10_col4" class="data row10 col4" >0.5398</td>
      <td id="T_6b964_row10_col5" class="data row10 col5" >0.6223</td>
      <td id="T_6b964_row10_col6" class="data row10 col6" >0.0000</td>
      <td id="T_6b964_row10_col7" class="data row10 col7" >0.0000</td>
      <td id="T_6b964_row10_col8" class="data row10 col8" >0.0920</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row11" class="row_heading level0 row11" >ada</th>
      <td id="T_6b964_row11_col0" class="data row11 col0" >Ada Boost Classifier</td>
      <td id="T_6b964_row11_col1" class="data row11 col1" >0.6848</td>
      <td id="T_6b964_row11_col2" class="data row11 col2" >0.7266</td>
      <td id="T_6b964_row11_col3" class="data row11 col3" >0.6848</td>
      <td id="T_6b964_row11_col4" class="data row11 col4" >0.7221</td>
      <td id="T_6b964_row11_col5" class="data row11 col5" >0.6956</td>
      <td id="T_6b964_row11_col6" class="data row11 col6" >0.2707</td>
      <td id="T_6b964_row11_col7" class="data row11 col7" >0.2786</td>
      <td id="T_6b964_row11_col8" class="data row11 col8" >0.1230</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row12" class="row_heading level0 row12" >gbc</th>
      <td id="T_6b964_row12_col0" class="data row12 col0" >Gradient Boosting Classifier</td>
      <td id="T_6b964_row12_col1" class="data row12 col1" >0.6777</td>
      <td id="T_6b964_row12_col2" class="data row12 col2" >0.6702</td>
      <td id="T_6b964_row12_col3" class="data row12 col3" >0.6777</td>
      <td id="T_6b964_row12_col4" class="data row12 col4" >0.6892</td>
      <td id="T_6b964_row12_col5" class="data row12 col5" >0.6690</td>
      <td id="T_6b964_row12_col6" class="data row12 col6" >0.1716</td>
      <td id="T_6b964_row12_col7" class="data row12 col7" >0.1857</td>
      <td id="T_6b964_row12_col8" class="data row12 col8" >0.2200</td>
    </tr>
    <tr>
      <th id="T_6b964_level0_row13" class="row_heading level0 row13" >dt</th>
      <td id="T_6b964_row13_col0" class="data row13 col0" >Decision Tree Classifier</td>
      <td id="T_6b964_row13_col1" class="data row13 col1" >0.5262</td>
      <td id="T_6b964_row13_col2" class="data row13 col2" >0.5046</td>
      <td id="T_6b964_row13_col3" class="data row13 col3" >0.5262</td>
      <td id="T_6b964_row13_col4" class="data row13 col4" >0.6230</td>
      <td id="T_6b964_row13_col5" class="data row13 col5" >0.5306</td>
      <td id="T_6b964_row13_col6" class="data row13 col6" >0.0122</td>
      <td id="T_6b964_row13_col7" class="data row13 col7" >0.0212</td>
      <td id="T_6b964_row13_col8" class="data row13 col8" >0.1240</td>
    </tr>
  </tbody>
</table>







## Business Intelligence Certification

For Business Intelligence certification, the required units include:
* Data Collection
* Data Exploration
* Data Validation
* Data Definition
* Data Construction
* Data Integration
* Dashbord Development

As of this case, I also believe they could be simplified, such as: 
* Data Collection
* Data Understanding
* Business-Centric Objective
* Dashboard Development
* Dashboard Deployment

Several steps, such as Data Integration, are omitted here, since in my view they align more closely with the responsibilities of data engineering rather than Business Intelligence.

### Business-Centric Objective

We have covered part of Data Collection and Understanding before, now comes to the most important aspect of business intelligence, that is define the objective constrained by business-centric. This part will be heavily inspired by the Google Looker Guide 'The Art of Telling Stories With Data' that ca be accessed [[2]](https://services.google.com/fh/files/misc/2110-pmk-theartoftellingstorieswith-datas-ebook-8-5x11-en-ck.pdf).

#### Choosing the Right Visualization


Who is your audience?
* Is it an executive who is interested in overall business performance? 
* Is it the sales team that wants to determine whether or not they will hit their quota for the quarter?
* Is it the marketing team that needs to determine if their social media and email campaigns are effective?

What is your goal?
* Are you building something that is referential, such as a report?  Or a visualization that’s meant to A CEO or CFO may want to understan how revenue this year compares to last year and the year before that.
* Are you creating something that is operational, like a dashboard that people will continually check to help direct their workflow?

What are you trying to show or say?
* Are you comparing data across different categories  (bar chart)? 
* Are you highlighting changes over time (line chart)?
* Are you presenting a picture of the full data set (scatterplot or histogram)? A clear understanding of all these factors will help you zero in on a visualization that effectively speaks to your audience.


#### Different Users Have Different Questions and Require Different Data







![Data Granularity](https://raw.githubusercontent.com/rezahabibi96/GitBook/refs/heads/main/articles/.resources/refreshment-course-for-data-science-and-business-intelligence-certification/diff_users.jpeg)



#### Give Context







![Context First](https://raw.githubusercontent.com/rezahabibi96/GitBook/refs/heads/main/articles/.resources/refreshment-course-for-data-science-and-business-intelligence-certification/give_context.jpeg)



#### Choosing The Right Chart







![Appropriate Chart](https://raw.githubusercontent.com/rezahabibi96/GitBook/refs/heads/main/articles/.resources/refreshment-course-for-data-science-and-business-intelligence-certification/right_chart.jpeg)



#### Use Case







![Example Case](https://raw.githubusercontent.com/rezahabibi96/GitBook/refs/heads/main/articles/.resources/refreshment-course-for-data-science-and-business-intelligence-certification/use_case.png)



#### PS

The example use case (BI dashboard) mentioned earlier [[3]](https://public.tableau.com/app/profile/kubra.dizlek/viz/TelcoChurnDashboard_17331624461710/Homepage). was built using Tableau. Within the Business Intelligence ecosystem, there are many tools available, ranging from closed-source to open-source solutions.

What I would like to emphasize is that tools are just tools. As practitioners, we have the freedom to choose any technology stack, as long as it enables us to deliver insights and effectively address the business problem. You can even rely on something as simple as Excel, or take advantage of emerging Generative AI/LLM-based tools, such as LIDA [[4]](https://microsoft.github.io/lida).


{% embed url="https://microsoft.github.io/lida/files/lidavid.mp4" %}

Ultimately, while tools will continue to evolve, the fundamentals of Business Intelligence remain the same.
