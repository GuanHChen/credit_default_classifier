import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

dd = pd.read_csv("UCI_Dataset.csv")
dd.drop(columns='ID', inplace=True)

new_cols = ['lim', 'sex', 'edu', 'mar', 'age', 'repay1', 'repay2', 'repay3', 'repay4',
            'repay5', 'repay6', 'bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6',
            'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6', 'default']

dd.columns = new_cols

"""
Binary encoding of SEX variable
Female = 0
Male = 1
"""

dd["sex"] = dd["sex"].replace({2:0})

"""
Ordinal encoding of EDUCATION
0 = Others, Unknown         280+123+51+14
1 = High School             4917
2 = University              14030
3 = Graduate School         10585
"""

mapping = {
    1: 3,  # Graduate school to 3
    2: 2,  # University to 2
    3: 1,  # High school to 1
    4: 0,  # Others to 0
    5: 0,  # Unknown to 0
    6: 0,  # Unknown to 0
    0: 0   # For some reason, there are 14 instances of 0. These are also remapped to 0.
}
dd['edu'] = dd['edu'].map(mapping)

"""
One-hot encoding of MARRIAGE
0 = Not sure what this is   54
1 = Married                 13659
2 = Single                  15964
3 = Other                   323

Single and Married will be one hot encoded
Values of 0 and for Other will be dropped. If someone is neither married or single,
the other category is implicit
"""

enc = OneHotEncoder(sparse_output=False)
enc_mar = enc.fit_transform(dd[['mar']])
enc_col = enc.get_feature_names_out(['mar'])
mar_df = pd.DataFrame(enc_mar, columns=enc_col)
mar_df = mar_df.iloc[:, [1,2]]

# Onehot encoded dataframe

dd_part1 = dd.iloc[:, :4]
dd_part2 = dd.iloc[:, 4:]
dd = pd.concat([dd_part1, mar_df, dd_part2], axis=1)
dd.drop(columns='mar', inplace=True)
dd = dd.rename(columns={'mar_1': 'married', 'mar_2': 'single'})

# Insert the encoded dataframe where the mar variable is
# Drop the mar column

"""
########################################
Feature Engineering
Credit utilization
637 Counts have utilization greater than 1
201 Counts have utilization less than 0 
########################################
"""

dd['util'] = dd[['bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6']].sum(axis=1) / (6 * dd['lim'])
# 269/838 outlier utils = 32%
# 6636/30000 Overall = 22%
# 218/637 Greater than 1 = 34% Statistically Significant!
# 51/201 Less than 0 = 25% Statistically Insig

repay_cols = ['repay1', 'repay2', 'repay3', 'repay4', 'repay5', 'repay6'] # list of the repayment variables
dd['late'] = (dd[repay_cols] > 0).sum(axis=1) # count of total late payments

bill_cols = ['bill1','bill2','bill3','bill4','bill5','bill6'] # list of the bill variables (spending)
dd['vol'] = dd[bill_cols].var(axis=1) # variance in spending

target = dd.pop('default')
dd['default'] = target # make default the last column

X = dd.drop('default', axis=1)
y = dd['default']

X_train, X_test, y_train, y_test = train_test_split( # 60-20-20 Split for Train, Validation, Test
    X, y, test_size=0.2, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

scaler = StandardScaler() # initialize scaler

scale_cols = ['lim','age',
              'bill1','bill2','bill3','bill4','bill5','bill6',
              'pay1','pay2','pay3','pay4','pay5','pay6',
              'util','late','vol']

ct = ColumnTransformer([
    ('scaler', StandardScaler(), scale_cols)
    ], remainder='passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
X_val = ct.transform(X_val)

X_train_df = pd.DataFrame(X_train)

"""
Modelling
Random Forest
                           
Normal Model: 
              precision    recall  f1-score   support
           0       0.84      0.94      0.89      4673
           1       0.65      0.37      0.47      1327
    accuracy                           0.82      6000
    
Hyperparameter Tuned:       
              precision    recall  f1-score   support
           0       0.84      0.96      0.89      4673
           1       0.70      0.36      0.48      1327
    accuracy                           0.82      6000                
"""

rf = RandomForestClassifier(random_state=123)
rf2 = RandomForestClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_leaf=10,
    min_samples_split=100,
    n_estimators=1500,
    random_state=123
)

#rf.fit(X_train, y_train) # fitting the model

rf2.fit(X_train, y_train)




y_pred = rf2.predict(X_test)

y_pred_proba = rf2.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

features = pd.DataFrame(rf2.feature_importances_, index=X_train_df.columns)

roc_auc = roc_auc_score(y_test, y_pred_proba)


print(features)
print(f"AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
"""
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Random Forest ROC')
plt.legend(loc="lower right")
plt.show()
"""


