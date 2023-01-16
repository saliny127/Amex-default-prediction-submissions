import numpy as np
import pandas as pd
import lightgbm
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, accuracy_score

import shap
import itertools

import seaborn as sns

import matplotlib.style as style
import seaborn as sns

import os

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier

import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train = pd.read_parquet(
    "/kaggle/input/amex-data-integer-dtypes-parquet-format/train.parquet")
test = pd.read_parquet(
    "/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet")
train_labels = pd.read_csv("../input/amex-default-prediction/train_labels.csv")


print(train.shape, test.shape, train_labels.shape)

train.duplicated().sum()

for col in train.columns.values:
    print(col, '-', train[col].isna().sum()/len(train[col]))

non_numeric_cols = train.columns[train.dtypes == 'object'].values
non_numeric_cols

numeric_cols = train.columns[train.dtypes != 'object'].values
numeric_cols

for i in numeric_cols:
    X = train[i].round(decimals=3)
    plt.figure(i)
    ax = sns.countplot(X)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=5)
    plt.tight_layout()
    plt.show()


style.use('seaborn-poster')
sns.set_style('ticks')
plt.subplots(figsize=(270, 200))

mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(train.corr(), cmap=plt.get_cmap('Blues'), annot=True, mask=mask, center=0, square=True,
            )

plt.title("Heatmap of all the Features", fontsize=25)


train['Date'] = pd.to_datetime(train['S_2'], format="%Y/%m/%d")
train['weekday'] = train['Date'].dt.weekday
train['day'] = train['Date'].dt.day
train['month'] = train['Date'].dt.month
train['year'] = train['Date'].dt.year

train['S_2'] = pd.to_numeric(train['S_2'].str.replace('-', ''))
print(train['S_2'])

train.drop(['Date'], axis=1, inplace=True)
print(train.shape)

cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117",
                "D_120", "D_126", "D_63", "D_64", "D_66", "D_68", 'customer_ID']

features = [col for col in train.columns.values if col not in cat_features]
features.append('customer_ID')

for i in cat_features:
    if train[i].dtype == 'int64':
        train.astype('int16')

train_cat = train[cat_features].groupby(
    'customer_ID', as_index=False).agg(['last', 'nunique'])
print(train_cat.shape)

for i in train.columns:
    if train[i].dtype == 'float64':
        train.astype('float16')

print(train.shape)

drop_features = ["B_30", "B_38", "D_114", "D_116", "D_117",
                 "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]

train.drop(drop_features, axis=1, inplace=True)

train = train.groupby('customer_ID', as_index=False).agg(
    ['mean', 'std', 'last'])

print(train.shape)

gc.collect()

print(train.columns.values)

train = train.merge(train_cat, how='inner', on="customer_ID")
print(train.shape)

del train_cat

join_col = []
for i in train.columns.values:
    if type(i) is tuple:
        col = '_'.join(i)
        join_col.append(col)
train.columns = join_col
train.reset_index()

train = train.merge(train_labels, how='inner', on="customer_ID")
print(train.shape)

del train_labels

corr_matrix = train.corr().abs()


upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


to_drop = [column for column in upper.columns if any(upper[column] >= 0.99)]


len(to_drop)


train.drop(to_drop, axis=1, inplace=True)


test['Date'] = pd.to_datetime(test['S_2'], format="%Y/%m/%d")
test['weekday'] = test['Date'].dt.weekday
test['day'] = test['Date'].dt.day
test['month'] = test['Date'].dt.month
test['year'] = test['Date'].dt.year

test['S_2'] = pd.to_numeric(test['S_2'].str.replace('-', ''))


test.drop(['Date'], axis=1, inplace=True)

cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117",
                "D_120", "D_126", "D_63", "D_64", "D_66", "D_68", 'customer_ID']

for i in test.columns.values:
    if test[i].dtype == 'int64' and i == 'customer_ID':
        test.astype('int16')

test_copy = test[cat_features].groupby(
    'customer_ID', as_index=False).agg(['last', 'nunique'])

for i in test.columns.values:
    if test[i].dtype == 'float64':
        test.astype('float16')

drp_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
           'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

test.drop(drp_col, axis=1, inplace=True)

gc.collect()

test = test.groupby('customer_ID', as_index=False).agg(['mean', 'std', 'last'])

test = test.merge(test_copy, how='inner', on="customer_ID")

join_col = []
for i in test.columns.values:
    if type(i) is tuple:
        col = '_'.join(i)
        join_col.append(col)

test.columns = join_col
test.reset_index()

test.drop(to_drop, axis=1, inplace=True)

del test_copy

gc.collect()


target = train['target']
Features = train.drop('target', axis=1, inplace=False)
print(train.shape, Features.shape)

numeric_cols = Features.columns[Features.dtypes != "object"].values
non_numeric_cols = Features.columns[Features.dtypes == 'object'].values

print(Features.shape, test.shape)

#feature engineering processes

#Scaling and Imputation
numeric_preprocessing_steps = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),
])

# imputation and Encoding
non_numeric_preprocessing_steps = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#transformation
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessing_steps, numeric_cols),
        # ("non_numeric",non_numeric_preprocessing_steps,non_numeric_cols)
    ],
    remainder='drop'
)

#splitting
X_train, X_eval, y_train, y_eval = train_test_split(
    Features,
    train['target'],
    test_size=0.2,
    shuffle=True,
    random_state=8
)


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

# KNN


KNN = KNeighborsClassifier(15)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", KNN),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

# SVM


svm = SVC(kernel='linear')

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", svm),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

# LightGBM

lgbm = LGBMClassifier(
    n_estimators=3000,
    num_leaves=100,
    learning_rate=0.01,
    colsample_bytree=0.6,
    objective='binary',
    max_depth=8,
    min_data_in_leaf=27,
    bagging_freq=7,
    bagging_fraction=0.8,
    feature_fraction=0.4,
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", lgbm),
])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict_proba(X_eval)

print(amex_metric(y_pred, y_eval))

# Output

full_pipeline.fit(Features, target)

joblib.dump(full_pipeline, 'pipes.joblib')

test_probas = full_pipeline.predict_proba(test)

test = pd.read_parquet(
    "/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet")

tests = test.groupby('customer_ID').tail(1)

print(tests.shape)

print(test_probas[:, 1])

tests['prediction'] = test_probas[:, 1]
print(tests['prediction'])

sub = tests[['customer_ID', 'prediction']]
sub.shape

sub.to_csv("my_submission.csv", index=False)

# Parameter tuning

parameters_gb = {
    'estimators__lgbm__n_estimators': [2000, 3000, 4000],
    'estimators__lgbm__max_depth': [7, 8],
    'estimators__lgbm__learning_rate': [0.01, 0.02],
    'estimators__lgbm__bagging_fraction': [0.6, 0.8],
    'estimators__lgbm__feature_fraction': [0.4, 0.6],
}

est_xgb = XGBClassifier()
est_lgbm = LGBMClassifier()

estimators_st = [
    ('lgbm', est_lgbm)
]


stacked_estimator = StackingClassifier(estimators=estimators_st,
                                       final_estimator=LogisticRegression,
                                       stack_method='predict_proba'
                                       )

full_pipeline_gs = GridSearchCV(estimator=Pipeline([
    ("preprocessor", preprocessor),
    ("estimators",   lgbm)
]),  param_grid=parameters_gb)


