# dev_bootstrap.py
# 開発用のおまじない。
# 対話型ウィンドウでファイルを実行したときに、
# 「プロジェクトルートをカレントディレクトリ＆import対象」にしてくれる。

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent  # = プロジェクトルート

# import 探す場所にルートを追加
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# カレントディレクトリもルートにしておく
try:
    os.chdir(ROOT)
except Exception:
    pass

# 開発中だけの表示。うざかったらコメントアウトしてOK。
print(f"[dev] ROOT set to: {ROOT}")
