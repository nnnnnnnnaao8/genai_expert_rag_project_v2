# このファイルは、RAGの性能を簡易に評価するためのスクリプトです。
# QAペアCSVを読み込み、RAGの回答がどの程度「想定回答」を含むかを計測します。

from pathlib import Path
from typing import List, Tuple

# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass


import csv

from config import QA_PAIRS_DIR
from .rag_baseline import answer


QA_FILE = QA_PAIRS_DIR / "sample_qa.csv"


def load_qa_pairs(path: Path) -> List[Tuple[str, str]]:
    """
    CSV から (question, answer) のペアを読み込む。
    列名: question, answer を想定。
    """
    pairs: List[Tuple[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"QAファイルが見つかりません: {path}")

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["question"], row["answer"]))
    return pairs


def simple_exact_match(pred: str, gold: str) -> bool:
    """
    超シンプルな指標: gold の文字列が pred に含まれているかどうか。
    """
    return gold.strip() in pred.strip()


def main():
    qa_pairs = load_qa_pairs(QA_FILE)
    correct = 0
    total = len(qa_pairs)

    for q, gold in qa_pairs:
        pred, meta = answer(q, k=5)
        if simple_exact_match(pred, gold):
            correct += 1

        print("Q:", q)
        print("PRED:", pred)
        print("GOLD:", gold)
        print("-" * 60)

    acc = correct / total if total > 0 else 0.0
    print(f"ExactMatch: {acc:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
