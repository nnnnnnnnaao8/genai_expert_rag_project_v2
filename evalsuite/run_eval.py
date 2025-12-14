# このファイルは、phase2_rag_core.eval_rag より少し汎用的に、
# QAペアに対して複数の指標(EM, F1)を計算する評価スクリプトです。
# 将来的にエージェント評価や別タスク評価にも拡張できるようにしてあります。

from pathlib import Path
from typing import List, Tuple
import csv

from config import QA_PAIRS_DIR
from evalsuite.metrics import exact_match, simple_f1
from phase2_rag_core.rag_baseline import answer


QA_FILE = QA_PAIRS_DIR / "sample_qa.csv"


def load_qa_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"QAファイルが見つかりません: {path}")
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["question"], row["answer"]))
    return pairs


def main():
    qa_pairs = load_qa_pairs(QA_FILE)

    em_sum = 0.0
    f1_sum = 0.0
    total = len(qa_pairs)

    for q, gold in qa_pairs:
        pred, meta = answer(q, k=5)
        em = exact_match(pred, gold)
        f1 = simple_f1(pred, gold)
        em_sum += em
        f1_sum += f1

        print("Q:", q)
        print("PRED:", pred)
        print("GOLD:", gold)
        print(f"EM: {em:.3f}, F1: {f1:.3f}")
        print("-" * 60)

    if total > 0:
        print(f"Mean EM : {em_sum / total:.3f}")
        print(f"Mean F1 : {f1_sum / total:.3f}")


if __name__ == "__main__":
    main()
