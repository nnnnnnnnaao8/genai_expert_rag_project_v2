# このファイルは、RAGに関する設定値（インデックス保存先・使用する埋め込みモデル名など）を集約するためのものです。
# RAGの挙動を調整したいときは、基本的にこのファイルの値を変えればOKです。

from config import INDEX_DIR, PDF_DIR

# ベクトルストアに用いる埋め込みモデル
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 検索時のデフォルトの top_k
DEFAULT_TOP_K = 5

# PDF と インデックスディレクトリもここから参照できるようにする
RAG_PDF_DIR = PDF_DIR
RAG_INDEX_DIR = INDEX_DIR
