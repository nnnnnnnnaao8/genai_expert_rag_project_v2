# このファイルは、RAG用インデックスの再構築やデータ更新など、「運用上のメンテナンス処理」をまとめるためのものです。
# 現場導入を意識した「知識ベースの更新フロー」のたたき台として使えます。

from phase2_rag_core.index_builder import build_index
# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass


def rebuild_all_indexes() -> None:
    """
    現時点では単に PDF からインデックスを再構築するだけ。
    今後、他のデータソース追加時にはここから呼び出す想定。
    """
    print("[INFO] RAG用インデックスの再構築を開始します。")
    build_index()
    print("[INFO] 再構築が完了しました。")


if __name__ == "__main__":
    rebuild_all_indexes()
