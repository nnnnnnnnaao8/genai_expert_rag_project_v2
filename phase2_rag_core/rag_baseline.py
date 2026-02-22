# このファイルは、最小構成のRAG検索＋回答生成ロジックを提供するためのものです。
# eval_rag や エージェントのツールから利用され、「RAGのベースライン実装」として機能します。
#　 質問(クエリ)に対する回答をindexから探索したうえで、生成します


# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



from config import client
from .rag_config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    RAG_INDEX_DIR,
)


def get_collection():
    """
    Chroma のコレクションを取得する。
     Chroma（クロマ：ベクトルDBライブラリ）は文章のembedding（埋め込みベクトル）を保存して、近いものを検索できるDB。
     collection（コレクション：DB内の“箱”）って何？
     →Chromaの中で「docs」みたいなデータのまとまり（名前付きの入れ物）。
      あなたのコードだと name="docs" がそれ。

    いつ出てくる？
    1.build_index() のとき：get_or_create_collection("docs") → 登録する箱を用意
    2.retrieve() のとき：同じ "docs" を取得 → 検索する
    """
    db = chromadb.PersistentClient(path=str(RAG_INDEX_DIR))
    collection = db.get_or_create_collection(
        name="docs",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        ),
    )
    return collection


def retrieve(query: str, k: int = DEFAULT_TOP_K):
    """
    クエリに対して関連チャンクを k 件取得する。
    """
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))


def build_prompt(query: str, contexts) -> str:
    """
    コンテキストと質問から、LLMに渡すプロンプトを組み立てる。(RAGでいうコンテキストはベクトルDBから拾ってきたチャンクのこと)
    """
    context_text = "\n\n".join(
        [f"[{i}] {c[0]}" for i, c in enumerate(contexts)]
    )
    prompt = f"""あなたは厳密で丁寧な日本語アシスタントです。
次のコンテキストを元に質問に答えてください。
コンテキストに情報がなければ「わかりません」と答えてください。

### コンテキスト
{context_text}

### 質問
{query}

### 回答
"""
    return prompt


def answer(query: str, k: int = DEFAULT_TOP_K) -> tuple[str, dict]:
    """
    質問に対してRAGで回答を生成し、テキストとメタ情報を返す。
    メタ情報とは、ページ数とか、文書名とか
    """
    contexts = retrieve(query, k=k)
    prompt = build_prompt(query, contexts)

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    content = chat.choices[0].message.content.strip()

    meta = {
        "contexts": contexts,
        "prompt_tokens": getattr(chat.usage, "prompt_tokens", None),
        "completion_tokens": getattr(chat.usage, "completion_tokens", None),
        "total_tokens": getattr(chat.usage, "total_tokens", None),
    }
    return content, meta
