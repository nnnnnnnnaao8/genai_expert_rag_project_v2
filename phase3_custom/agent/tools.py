# このファイルは、エージェントが呼び出す「ツール」を定義するためのものです。
# RAG検索や、CSV集計などの処理を関数として切り出し、エージェントから利用できるようにします。

import pandas as pd
# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass

from config import RAW_DATA_DIR
from phase2_rag_core.rag_baseline import answer as rag_answer


def tool_rag_search(query: str) -> str:
    """
    RAGベースで質問に答えるツール。
    """
    resp, meta = rag_answer(query, k=5)
    return resp


def tool_csv_summary(filename: str = "sample_regression.csv") -> str:
    """
    CSVファイルの統計サマリを返すツール。
    実務では品質データや検査データの概要確認などを想定。
    """
    path = RAW_DATA_DIR / filename
    if not path.exists():
        return f"{path} が見つかりません。"

    df = pd.read_csv(path)
    desc = df.describe(include="all").to_string()
    return f"データ概要:\n{desc}"
