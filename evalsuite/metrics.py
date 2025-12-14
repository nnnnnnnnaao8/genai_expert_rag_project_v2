# このファイルは、RAGやQAシステムの性能を測るための指標（EM, F1 など）を定義するためのものです。
# まずは単純な Exact Match と、トークン単位F1の簡易版を実装しています。

from typing import Tuple


def exact_match(pred: str, gold: str) -> float:
    """
    gold が pred に含まれていれば 1.0、そうでなければ 0.0 を返す。
    """
    return 1.0 if gold.strip() in pred.strip() else 0.0


def simple_f1(pred: str, gold: str) -> float:
    """
    非常に簡略化したF1スコア。
    トークン分割して、共通トークンのPrecision/RecallからF1を計算する。
    """
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    if not pred_set or not gold_set:
        return 0.0

    inter = pred_set & gold_set
    precision = len(inter) / len(pred_set)
    recall = len(inter) / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
