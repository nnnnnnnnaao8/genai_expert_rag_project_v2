# このファイルは、RAGやQAシステムの性能を測るための指標（EM, F1 など）を定義するためのものです。
# まずは単純な Exact Match と、トークン単位F1の簡易版を実装しています。

from typing import Tuple

#-------------------n-gram追加-----------------------
from collections import Counter
import re

def _normalize_text(s: str) -> str:
    """
    軽い正規化（normalization：表記ゆれ吸収）
    - 空白除去
    - 記号の一部除去
    """
    s = s.strip()
    s = re.sub(r"\s+", "", s)  # 全ての空白を消す
    return s

def _char_ngrams(s: str, n: int = 2):
    """
    文字n-gramを作る（例：n=2なら「照度」「度は」「は5」...）
    """
    s = _normalize_text(s)
    if len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def char_ngram_f1(pred: str, gold: str, n: int = 2) -> float:
    """
    文字n-gram版F1（日本語でも動く簡易F1）
    - pred/gold を n文字ずつのかたまりにして重なりからF1計算
    """
    pred_ngrams = _char_ngrams(pred, n=n)
    gold_ngrams = _char_ngrams(gold, n=n)

    if not pred_ngrams or not gold_ngrams:
        return 0.0

    pred_cnt = Counter(pred_ngrams)
    gold_cnt = Counter(gold_ngrams)

    # 共通部分（重なり数）
    inter = sum((pred_cnt & gold_cnt).values())
    if inter == 0:
        return 0.0

    precision = inter / sum(pred_cnt.values())
    recall = inter / sum(gold_cnt.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

#--------↑n-gram追加


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
