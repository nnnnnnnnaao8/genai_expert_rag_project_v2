# このファイルは、phase0_data_pipeline フォルダで利用する共通のデータ読み込み
# ＆簡単な前処理ロジックを定義するためのものです。
# 主に CSV 読み込み、簡易クリーニング、特徴量と目的変数の分割を行います。


# """
# このファイルの設計思想（超重要）
# ・モデルに前処理責任を押し付けない
# ・回帰・分類・将来の別モデルでも再利用可能
# ・「壊れたらすぐ落ちる」安全設計
# -------------------------------
# 面接で言える要約（そのまま使える）

# load_data.py では、生データをモデルが扱える形に整える責務を分離しています。
# これにより前処理と学習ロジックを疎結合にし、
# 回帰・分類の両タスクで再利用できる構成にしています。

# """

from typing import Tuple
import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

#PROCESSED_DATA_DIR(前処理後のデータの保存先)
#RAW_DATA_DIR （生データの場所）
#これはconfig.pyに入っている変数である。
def load_raw_csv(filename: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    RAW_DATA_DIR から CSV を読み込む。
    filename:str →"sample.csv"みたいなファイル名だけを渡す設計
    encoding(文字コード):str →日本語csv対応
    pd.DataFrame → 戻り値(表)
    """
    path = RAW_DATA_DIR / filename#ここで、failenameが文字列でないと、Pathと結合できず型エラー(TypeError)になる
    """
    Path / "file.csv" は pathlib の書き方
    →config.py で Path を使って作った RAW_DATA_DIR
    それを import している
    ⇒Pathオブジェクトがそのまま渡ってきている
    """
    if not path.exists():#パスの存在でtrue or False 
        raise FileNotFoundError(f"{path} が存在しません。")
        #なぜ、raise?⇒意味は一言で、ここで処理を強制終了して、エラーとして伝えろ
        #printだと、、、

        #print("ないよ")
        #return None
        #⇒皇族処理がNoneで動いて謎エラーになる

    return pd.read_csv(path, encoding=encoding)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:# 入力DataFrame、出力DataFrameで
                                                    #DataFrameを加工している
    """
    超シンプルな前処理：
    - 重複行の削除
    - 文字列列の前後スペース除去
    - 全欠損の列削除
    """
    df = df.drop_duplicates()#データフレームの重複行削除 実務でかなり頻出

    #文字列だけ処理するフィルタ↓
    for col in df.select_dtypes(include=["object"]).columns:#object型とは文字列型のこと
        #↓astype(str):NaNや数値が混ざっていても強制文字列化
        #str.strip()：前後の空白削除 " A "→"A"
        df[col] = df[col].astype(str).str.strip()
    
    #↓axis=1：列方向
    #↓how="all"：全部NaNの列だけ削除
    df = df.dropna(axis=1, how="all")
    return df      #DataFrameを返す

    
    # 【参考】
    # object型 = 文字列型？ なぜ str と言わない？
    # →pandas の世界では…
    # ・str は Pythonの型
    # ・object は pandasの列型（dtype）
    
    # 例えば、df.dtypes を出力すると
    # age        int64
    # gender     object
    
    # ......なぜ object？
    # pandasは昔から
    # ・文字列
    # ・混在型（str + None + 数値）
    #   を 全部 object として扱ってきた歴史がある。
    # 👉 「文字列っぽい列」＝ object
    # ※ 最近は string dtype もあるが、
    # 実務ではまだ object が多い。
                                        

def split_features_target(
    df: pd.DataFrame, target_col: str #関数への入力 df：全データ target_col：目的変数名
) -> Tuple[pd.DataFrame, pd.Series]: #Tuple:複数の値を1セットで返す型(X,y)みたいなやつ
                                     # X:説明変数(DataFrame) Y:目的変数(Series)
                                     #Tuple[A, B]⇒1番目はA型、2番目はB型 で分かれている。
    """
    特徴量 X と目的変数 y に分割するってだけ。目的変数名が分かってればそれで分けれる。
    X, y = split_features_target(...)
    これをするとき、タプルを返す関数であればできる

    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} が列に含まれていません。")#列名ミス即検出 →面接で評価
    X = df.drop(columns=[target_col])#Xは説明変数なので、目的変数以外の列をdropする
    y = df[target_col]
    return X, y


def save_processed(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    #index=False CSVに行番号を保存しない(実務ではほぼこれ)
    """
    前処理済みデータを PROCESSED_DATA_DIR 以下に保存する。
    """
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=index)  #CSVで保存
