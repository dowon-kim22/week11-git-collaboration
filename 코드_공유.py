# 코드_공유.py
# 작업자 A, B, Admin 이 함께 사용하는 파일입니다.

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

# 2013년 뉴욕 출발 항공편 표본 50,000건입니다.
# 항공편 정보에 기상 정보와 기체 정보를 미리 결합해 두었습니다.
df = pd.read_csv('flights_ml.csv')

NUM = ['month', 'day', 'sched_hour', 'distance', 'temp', 'humid',
       'wind_speed', 'precip', 'visib', 'seats', 'engines']
CAT = ['carrier', 'origin']

X = pd.get_dummies(df[NUM + CAT], columns=CAT)
X = X.fillna(X.median(numeric_only=True))
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

# 1. DecisionTree 하이퍼파라미터 후보 설정
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 7],
    'min_samples_leaf': [1, 5, 20]
}

# 2. GridSearchCV로 최적 하이퍼파라미터 탐색
dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

# 3. 최적 파라미터 및 성능 확인
dt_model = dt_grid.best_estimator_
dt_pred = dt_model.predict(X_test)

print("DecisionTree 최적 파라미터:", dt_grid.best_params_)
print("DecisionTree F1 score:", f1_score(y_test, dt_pred))
print("DecisionTree Accuracy:", accuracy_score(y_test, dt_pred))

# 4. Feature Importance 상위 10개 시각화
dt_importance = pd.Series(
    dt_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
dt_importance.sort_values().plot(kind='barh')
plt.title('DecisionTree Feature Importance Top 10')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

from xgboost import XGBClassifier

# 1. XGBoost 하이퍼파라미터 후보 설정
xgb_param_grid = {
    'max_depth': [3, 6, 9],
    'n_estimators': [100, 300],
    'learning_rate': [0.1, 0.05]
}

# 2. GridSearchCV로 최적 하이퍼파라미터 탐색
xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=1
    ),
    param_grid=xgb_param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

# 3. 최적 파라미터 및 성능 확인
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)

print("XGBoost 최적 파라미터:", xgb_grid.best_params_)
print("XGBoost F1 score:", f1_score(y_test, xgb_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

# 4. Feature Importance 상위 10개 시각화
xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
xgb_importance.sort_values().plot(kind='barh')
plt.title('XGBoost Feature Importance Top 10')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()
