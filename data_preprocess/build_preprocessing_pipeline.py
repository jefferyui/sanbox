import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import TransformerMixin, BaseEstimator

# 自訂轉換器：處理極端值（以 IQR 為例）
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        # 計算每欄的上下四分位數
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.lower_ = Q1 - self.factor * (Q3 - Q1)
        self.upper_ = Q3 + self.factor * (Q3 - Q1)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped

# 主流程函式
def build_preprocessing_pipeline(
    df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    target_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    1. 切分訓練/測試
    2. 數值欄位：缺值填補、去除常數欄、去極端值、縮放
    3. 類別欄位：缺值填補、編碼
    4. 合併所有前處理步驟
    """
    # 1. 切分
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X, y = df, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. 數值欄位子流程
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),       # 缺值填補
        ('clipper', OutlierClipper(factor=1.5)),              # 極端值處理
        ('scaler', StandardScaler()),                        # 標準化；可改 MinMaxScaler()
        ('var_thresh', VarianceThreshold(threshold=0.0)),     # 去除常數欄
    ])

    # 3. 類別欄位子流程
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        # 若類別順序有意義，可改用 OrdinalEncoder()
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    # 4. 合併
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ], remainder='drop')

    # 5. 建立最終整合 Pipeline（含模型也可串在這裡）
    full_pipeline = Pipeline([
        ('preproc', preprocessor),
        # ('clf', YourModel(...)),
    ])

    return full_pipeline, X_train, X_test, y_train, y_test

# 範例使用方式
if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")
    numeric_cols = ["age", "salary", "years_experience"]
    categorical_cols = ["gender", "department", "region"]
    pipeline, X_tr, X_te, y_tr, y_te = build_preprocessing_pipeline(
        df, numeric_cols, categorical_cols, target_col="target"
    )

    # 只做前處理
    X_tr_prep = pipeline.fit_transform(X_tr)
    X_te_prep = pipeline.transform(X_te)
    print("前處理後訓練資料維度：", X_tr_prep.shape)


數值欄位處理

缺值填補：以中位數或平均數填補，避免受極端值影響。

極端值處理：利用 IQR 裁切（Clip）或 Winsorization，防止異常值干擾模型。

特徵縮放：標準化（StandardScaler）或最小最大值縮放（MinMaxScaler），讓梯度下降更穩定。

去除常數欄：拿掉方差為零的欄位，減少冗餘。


