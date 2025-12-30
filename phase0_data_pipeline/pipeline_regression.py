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

from phase0_data_pipeline.load_data import basic_clean, load_raw_csv, split_features_target
#from phase0_data_pipeline.load_data import basic_clean, load_raw_csv, split_features_target

# """
# 標準化は「テストデータにも行うのか？」
# 結論
# 👉 行う。ただし「やり方」が重要。

# 正しい流れ（超重要）
#  訓練データ（train data） 
#  平均・分散を計算する（これが fit）
#  テストデータ（test data）
#  訓練データで計算した平均・分散を使って変換する（これが transform）

# ❌ やってはいけないこと
#  テストデータ単体で平均・分散を計算する
#  → データリーク（意味：評価データの情報が学習に漏れる）

# """


#どんな列が数値で、どんな列がカテゴリかを受け取り、それに応じた前処理＋モデルを組み立てる関数
def build_regression_pipeline(
    numeric_features, categorical_features #それぞれ列名のリスト(量的特徴列とカテゴリ列の名前)
) -> Pipeline:
    """
    数値＋カテゴリ列をまとめて扱う回帰パイプラインを組み立てる。
    """
   
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),#StandardScaler()標準化
            # "scaler"：ラベル。任意の名前（文字列）
            # 人間が「このステップは何か」を識別するため
        ]
    )

    categorical_transformer = Pipeline(  
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    #パイプラインを作成した、numeric_transformerと、categorical_transformerをそれぞれ前処理
    preprocessor = ColumnTransformer(  #ColumnTransformer：前処理したい(transformerしたい)特定の列に適用
        transformers=[
            ("num", numeric_transformer, numeric_features),#量的変数に標準化
            ("cat", categorical_transformer, categorical_features),#カテゴリ化
        ]
    )

    model = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),#すでに、量とカテゴリ列の前処理した変数preprocessor
            ("model", model),            #モデル
        ]
    )
    return pipe

#CSV1つを渡せば、学習から評価まで一気にやる」関数
def train_and_eval(
    filename: str,
    target_col: str,#目的変数列名 だからstr
    test_size: float = 0.2,
    random_state: int = 42,#random_state（乱数の種：ランダムの固定値）を固定すると分割結果が毎回同じになるため指定 
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
    df = basic_clean(df)   #load_data.pyで作った重複行削除や欠陥削除など基本的な前処理
    X, y = split_features_target(df, target_col) #target_col(目的変数の列)をyにする　それ以外をX(特徴量)にする
    
    #自動で、量的特徴列と、カテゴリ列を分ける処理　
    #↓Xのうち、tolist()：リストに入れる、columnsを、データ型(dtype:列の型)を見てnp.numberを含むものを numeric_featuresに 
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    #↑説明変数Xが量的変数でないもの(カテゴリカルなもの)の列はcategorical_featuresに代入

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    #もう一つ一番上で作った関数を適用⇒pipeには、前処理した量とカテゴリ列と使用するモデルを入れたパイプラインが代入される
    pipe = build_regression_pipeline(numeric_features, categorical_features)
    #fit（学習）時に：
    # scaler は train のみで統計量を計算
    # test データにはその統計量を使う
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)#predict fitで学習したモデルを用いて予測
    #rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")


if __name__ == "__main__":#このファイルを「直接実行したときだけ」動く
    # ∟ import（読み込み）された時は動かない
    # ∟ スト実行・単体実行に便利

    # 実際の使い方：
    # data/raw/sample_regression.csv を置き、目的変数名を指定する
    train_and_eval(filename="sample_regression.csv", target_col="target")
    #sample_regression.csv sample_regression_income_as_text
