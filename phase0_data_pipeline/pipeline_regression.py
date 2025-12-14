# このファイルは、回帰タスク向けの機械学習パイプライン（前処理→学習→評価）を構築するためのものです。
# 学習済みエンジニアとして「前処理と分析パイプラインを設計できる」ことを証明する用途を想定しています。

from typing import Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .load_data import basic_clean, load_raw_csv, split_features_target


def build_regression_pipeline(
    numeric_features, categorical_features
) -> Pipeline:
    """
    数値＋カテゴリ列をまとめて扱う回帰パイプラインを組み立てる。
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def train_and_eval(
    filename: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    一連の流れ：
    - CSVを読み込み
    - 簡易前処理
    - 特徴量と目的変数の分割
    - train/test 分割
    - パイプライン構築・学習
    - RMSE/MAE を出力
    """
    df = load_raw_csv(filename)
    df = basic_clean(df)
    X, y = split_features_target(df, target_col)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_regression_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")


if __name__ == "__main__":
    # 実際の使い方：
    # data/raw/sample_regression.csv を置き、目的変数名を指定する
    train_and_eval(filename="sample_regression.csv", target_col="target")
