import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import optuna
import matplotlib.pyplot as plt



def report(model):
    # Generates the feature importances and classification report of a model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    features = pd.DataFrame(model.feature_importances_, index=ct.get_feature_names_out())
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(features)
    print(f"AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def objective(trial):
    # Optuna hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 10, 1000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 500)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   criterion='entropy',
                                   n_jobs=-1,
                                   random_state=123)

    score = cross_val_score(model, X_val, y_val, cv=5, scoring='recall', n_jobs=-1)
    return score.mean()

def optimize():
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=123))
    study.optimize(objective, n_trials=200)
    best_params = study.best_params
    print(best_params)


dd = pd.read_csv("UCI_Dataset.csv")
dd.drop(columns='ID', inplace=True)

new_cols = ['lim', 'sex', 'edu', 'mar', 'age', 'repay1', 'repay2', 'repay3', 'repay4',
            'repay5', 'repay6', 'bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6',
            'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6', 'default']

dd.columns = new_cols

dd["sex"] = dd["sex"].replace({2:0})

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

ct = ColumnTransformer(
    [('scaler', StandardScaler(), scale_cols)],
    remainder='passthrough',
    verbose_feature_names_out=False
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
X_val = ct.transform(X_val)

rf = RandomForestClassifier(random_state=123)

rf2 = RandomForestClassifier(
    criterion='entropy',
    max_depth=41,
    min_samples_leaf=2,
    min_samples_split=115,
    n_estimators=1086,
    random_state=123
)

#rf2.fit(X_train, y_train)

#report(rf2)

#optimize()





