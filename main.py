import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.inspection import PartialDependenceDisplay
import optuna
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt


def pdp(feature):
    display = PartialDependenceDisplay.from_estimator(
        estimator=rf2,
        X=X_train,
        features=[feature],
        kind='both',
        centered=True,
        feature_names=ct.get_feature_names_out(),
        line_kw={'linewidth': 3, 'linestyle': '-'},
        ice_lines_kw={'alpha': 0.3, 'linewidth': 0.5, 'color': 'lightsteelblue'}
    )

    plt.suptitle(f'Partial Dependence and ICE Plot for {feature.title()} Payments')  # Add a main title
    display.axes_[0, 0].set_ylabel('Increased Probability of Default')  # Set y-axis label
    display.axes_[0, 0].set_xlabel('Late Payments')  # Set y-axis label
    ax = display.axes_[0, 0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    plt.show()



def heatmap():
    plot_cols = ['Limit', 'Sex', 'Edu', 'Mar', 'Age', 'Repay1', 'Repay2', 'Repay3', 'Repay4',
                 'Repay5', 'Repay6', 'Bill1', 'Bill2', 'Bill3', 'Bill4', 'Bill5', 'Bill6',
                 'Pay1', 'Pay2', 'Pay3', 'Pay4', 'Pay5', 'Pay6', 'Default']

    plot_dd.columns = plot_cols

    correlation_matrix = plot_dd.corr().abs()
    cols = correlation_matrix.columns.tolist()
    cols_reversed = cols[::-1]
    correlation_matrix = correlation_matrix[cols_reversed]

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix,
                annot=False,
                cmap='Oranges',
                fmt='.2f',
                cbar_kws={'label': 'Magnitude of Correlation'})
    plt.title('Correlation Heatmap of Predictors', fontsize=20)
    plt.tight_layout()
    plt.show()


def report(model):
    # Generates the feature importances and classification report of a model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    features = pd.DataFrame({'importance': model.feature_importances_}, index=ct.get_feature_names_out()).sort_values(
        by='importance', ascending=False)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(features)
    print(f"AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(brier_score_loss(y_test, y_pred_proba))


def obj1(trial):
    # Optuna hyperparameter optimization for forest
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 1000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 500)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   criterion='entropy',
                                   n_jobs=-1,
                                   random_state=123)

    score = cross_val_score(model, X_val, y_val, cv=5, scoring='f1', n_jobs=-1)
    return score.mean()


def obj2(trial):
    # Optuna hyperparameter optimization for XGBoost
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    eta = trial.suggest_float('eta', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 50)
    gamma = trial.suggest_int('gamma', 0, 10)
    alpha = trial.suggest_int('alpha', 0, 10)
    reg_lambda = trial.suggest_int('reg_lambda', 0, 10)
    max_delta_step = trial.suggest_int('max_delta_step', 0, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 0, 30)
    scale_pos_weight = 23364 / 6636

    model = xgb.XGBClassifier(n_estimators=n_estimators,
                              eta=eta,
                              max_depth=max_depth,
                              gamma=gamma,
                              alpha=alpha,
                              reg_lambda=reg_lambda,
                              max_delta_step=max_delta_step,
                              min_child_weight=min_child_weight,
                              scale_pos_weight=scale_pos_weight,
                              eval_metric='aucpr',
                              random_state=123)

    score = cross_val_score(model, X_val, y_val, cv=5, scoring='roc_auc', n_jobs=-1)
    return score.mean()


def optimize(obj):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(obj, n_trials=200)
    best_params = study.best_params
    print(best_params)


dd = pd.read_csv("UCI_Dataset.csv")
dd.drop(columns='ID', inplace=True)

new_cols = ['lim', 'sex', 'edu', 'mar', 'age', 'repay1', 'repay2', 'repay3', 'repay4',
            'repay5', 'repay6', 'bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6',
            'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6', 'default']

dd.columns = new_cols
plot_dd = dd

dd["sex"] = dd["sex"].replace({2: 0})

mapping = {
    1: 3,  # Graduate school to 3
    2: 2,  # University to 2
    3: 1,  # High school to 1
    4: 0,  # Others to 0
    5: 0,  # Unknown to 0
    6: 0,  # Unknown to 0
    0: 0  # For some reason, there are 14 instances of 0. These are also remapped to 0.
}

dd['edu'] = dd['edu'].map(mapping)

enc = OneHotEncoder(sparse_output=False)
enc_mar = enc.fit_transform(dd[['mar']])
enc_col = enc.get_feature_names_out(['mar'])
mar_df = pd.DataFrame(enc_mar, columns=enc_col)
mar_df = mar_df.iloc[:, [1, 2]]

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

repay_cols = ['repay1', 'repay2', 'repay3', 'repay4', 'repay5', 'repay6']  # list of the repayment variables
dd['late'] = (dd[repay_cols] > 0).sum(axis=1)  # count of total late payments

bill_cols = ['bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6']  # list of the bill variables (spending)
dd['vol'] = dd[bill_cols].var(axis=1)  # variance in spending

#target = dd.pop('default')
#dd['default'] = target  # make default the last column

X = dd.drop('default', axis=1)
y = dd['default']

X_train, X_test, y_train, y_test = train_test_split(  # 60-20-20 Split for Train, Validation, Test
    X, y, test_size=0.2, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

scaler = StandardScaler()  # initialize scaler

scale_cols = ['lim', 'age',
              'bill1', 'bill2', 'bill3', 'bill4', 'bill5', 'bill6',
              'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6',
              'util', 'vol'
              ]

ct = ColumnTransformer(
    [('scaler', StandardScaler(), scale_cols)],
    remainder='passthrough',
    verbose_feature_names_out=False
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
X_val = ct.transform(X_val)

# Robustness Check

#nr = X.drop('late', axis=1)

#nr_train, nr_test, y_train, y_test = train_test_split(  # 60-20-20 Split for Train, Validation, Test
#    nr, y, test_size=0.2, random_state=123, stratify=y)
#nr_train, nr_val, y_train, y_val = train_test_split(
#    nr_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

#nr_train = ct.fit_transform(nr_train)
#nr_test = ct.transform(nr_test)




rf1 = RandomForestClassifier(
    criterion='entropy',
    n_jobs=-1,
    random_state=123
)

rf2 = RandomForestClassifier(
    criterion='entropy',
    max_depth=31,
    min_samples_leaf=27,
    min_samples_split=123,
    n_estimators=1809,
    random_state=123,
    n_jobs=-1
)

rf3 = RandomForestClassifier(
    criterion='entropy',
    max_depth=41,
    min_samples_leaf=2,
    min_samples_split=115,
    n_estimators=1086,
    random_state=123,
    n_jobs=-1
)

bm1 = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    eta=0.1,
    max_depth=50,
    eval_metric='aucpr',
    random_state=123,
    n_jobs=-1
)

bm2 = xgb.XGBClassifier(objective='binary:logistic', # Or 'multi:softmax' for multi-class
                       n_estimators=1359,          # Number of boosting rounds (trees)
                       eta=0.09796,                # Step size shrinkage to prevent overfitting
                       max_depth=32,               # Maximum depth of a tree
                       gamma=9,
                       alpha=8,
                       reg_lambda=6,
                       max_delta_step=1,
                       min_child_weight=16,
                       eval_metric='aucpr',
                       random_state=123,
                       n_jobs=-1
)

bm3 = xgb.XGBClassifier(objective='binary:logistic', # Or 'multi:softmax' for multi-class
                       n_estimators=1359,          # Number of boosting rounds (trees)
                       eta=0.09796,                # Step size shrinkage to prevent overfitting
                       max_depth=32,               # Maximum depth of a tree
                       #gamma=9,
                       #alpha=8,
                       #reg_lambda=6,
                       max_delta_step=1,
                       min_child_weight=16,
                       eval_metric='aucpr',
                       random_state=123,
                       n_jobs=-1
)

start = time.time()

#rf1.fit(X_train, y_train)
rf2.fit(X_train, y_train)
#rf3.fit(X_train, y_train)
#bm1.fit(X_train, y_train)
#bm2.fit(X_train, y_train)
#bm3.fit(X_train, y_train)

end = time.time()
print(end-start)

#report(rf1)
#report(rf2)
#report(rf3)
#report(bm1)
#report(bm2)
#report(bm3)

pdp('late')
