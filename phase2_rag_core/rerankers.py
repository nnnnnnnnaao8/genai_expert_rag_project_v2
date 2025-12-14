# このファイルは、RAGの「再ランク（rerank）」機能を差し込むための拡張ポイントを定義するものです。
# 最初はダミー実装にしておき、後からスコア計算や別モデルによる再ランクを追加できるようにしてあります。

from typing import List, Tuple, Any


def identity_rerank(
    contexts: List[Tuple[str, dict]], query: str
) -> List[Tuple[str, dict]]:
    """
    何も再ランクを行わず、そのまま返すダミー関数。
    後でスコアリングや並び替えを実装したい場合は、ここを書き換える。
    """
    return contexts


# 将来追加用の例（実装は空のまま置いておく）
def cosine_score_rerank(
    contexts: List[Tuple[str, dict]], query: str
) -> List[Tuple[str, dict]]:
    """
    TODO: 埋め込みコサイン類似度で再ランクする実験用の関数。
    今のところ未実装で、identity_rerank を使う想定。
    """
    return contexts
