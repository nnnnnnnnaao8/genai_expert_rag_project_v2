# このファイルは、PDFからテキストを抽出し、チャンク分割→埋め込み→ベクトルDB(Chroma)への登録を行うRAGの前処理ステップを担当しています。
# 「RAGの構造理解（文書前処理〜インデックス構築）」を示すための中心的なスクリプトです。




from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import PDF_DIR
from phase1_basics.etl import read_pdf_text
from .rag_config import EMBEDDING_MODEL_NAME, RAG_INDEX_DIR



def simple_chunk(text: str, max_chars: int = 500) -> List[str]:
    """
    文字数ベースのシンプルなチャンク分割。
    """
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i : i + max_chars])
    return chunks


def build_index() -> None:
    """
    data/pdf 配下のPDFをすべて読み込み、チャンク生成→埋め込み→Chromaへ登録する。
    """
    client = chromadb.PersistentClient(path=str(RAG_INDEX_DIR))
    collection = client.get_or_create_collection(
        name="docs",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        ),
    )

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] PDFが見つかりません: {PDF_DIR}")
        return

    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[dict] = []

    for pdf in pdf_files:
        text = read_pdf_text(str(pdf))
        chunks = simple_chunk(text)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_ids.append(f"{pdf.name}_{idx}")
            all_metas.append({"source": pdf.name, "chunk_id": idx})

    collection.add(documents=all_texts, ids=all_ids, metadatas=all_metas)
    print(f"[INFO] インデックス構築完了。チャンク数: {len(all_texts)}")


if __name__ == "__main__":
    build_index()
