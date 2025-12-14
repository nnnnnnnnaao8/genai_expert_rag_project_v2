# このファイルは、分類タスク向けの機械学習パイプライン（前処理→学習→評価）を構築するためのものです。
# 線形回帰とは別に、分類問題でも前処理〜評価まで設計できることを示す用途を想定しています。


# sklearn  ← パッケージ（大きい箱）
#  ├─ compose            ← モジュール（小さい箱）
#  │    └─ ColumnTransformer
#  ├─ preprocessing       ← モジュール
#  │    ├─ OneHotEncoder
#  │    └─ StandardScaler
#  ├─ model_selection     ← モジュール
#  │    └─ train_test_split
#  ├─ metrics
#  │    └─ classification_report
#  └─ linear_model
#       └─ LogisticRegression

#**「前処理（cleaning） → 特徴量変換（feature engineering） → 学習（training） → 評価（evaluation）」
#を行っている。

# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass


import numpy as np
from sklearn.compose import ColumnTransformer #列ごとに前処理変える仕組み(標準化、文字ならOneHotEncoder)
from sklearn.linear_model import LogisticRegression #分類に使うロジスティック回帰モデル 最終的に予測predictionを行う
from sklearn.metrics import classification_report #モデルの良し悪し評価用の関数　precision recall f1-scoreをだす
from sklearn.model_selection import train_test_split #trainとtestを分割するためのやつ
from sklearn.pipeline import Pipeline #処理の一連の流れ（前処理→学習→予測）をひとつの“流れ作業”にする道具。
from sklearn.preprocessing import OneHotEncoder, StandardScaler #OneHotEncoder:カテゴリ変数変換 男と女を1と0にするとか、StandardScaler(標準化)



# # # プロジェクトのルートにいる前提なら本当は不要だが、念のための確認にも使える
# import os
# print(os.getcwd())

# from phase0_data_pipeline.pipeline_classification import train_and_eval
# train_and_eval(filename="sample_classification.csv", target_col="label")

#from .load_data import basic_clean, load_raw_csv, split_features_target

from phase0_data_pipeline.load_data import basic_clean, load_raw_csv, split_features_target



def build_classification_pipeline(
    numeric_features, categorical_features
) -> Pipeline:
    """
    数値＋カテゴリ列をまとめて扱う分類パイプラインを組み立てる。
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

    model = LogisticRegression(max_iter=1000)

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
    - classification_report を出力
    """
    df = load_raw_csv(filename)
    df = basic_clean(df)
    X, y = split_features_target(df, target_col)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_classification_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # 実際の使い方：
    # data/raw/sample_classification.csv を置き、目的変数名を指定する
    train_and_eval(filename="sample_classification.csv", target_col="label")
