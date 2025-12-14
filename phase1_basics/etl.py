# このファイルは、phase1_basics および phase2_rag_core から共通利用する「PDF→テキスト抽出」処理をまとめたものです。
# PDFファイルを読み込み、ページごとのテキストを結合して返します。

from pathlib import Path
from typing import List

from pypdf import PdfReader

from config import PDF_DIR


def read_pdf_text(path: str) -> str:
    """
    単一PDFからテキストを抽出して結合する。
    `path` が相対パスの場合は data/pdf 配下を基準とする。
    """
    pdf_path = Path(path)
    if not pdf_path.is_absolute():
        pdf_path = PDF_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF ファイルが見つかりません: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        texts: List[str] = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        raise RuntimeError(f"PDF読み取りに失敗しました: {e}")
