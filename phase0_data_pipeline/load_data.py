# このファイルは、phase0_data_pipeline フォルダで利用する共通のデータ読み込み
# ＆簡単な前処理ロジックを定義するためのものです。
# 主に CSV 読み込み、簡易クリーニング、特徴量と目的変数の分割を行います。

from typing import Tuple

import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_csv(filename: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    RAW_DATA_DIR から CSV を読み込む。
    """
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} が存在しません。")
    return pd.read_csv(path, encoding=encoding)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    超シンプルな前処理：
    - 重複行の削除
    - 文字列列の前後スペース除去
    - 全欠損の列削除
    """
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(axis=1, how="all")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    特徴量 X と目的変数 y に分割する。
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} が列に含まれていません。")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def save_processed(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """
    前処理済みデータを PROCESSED_DATA_DIR 以下に保存する。
    """
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=index)
