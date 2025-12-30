# このファイルは、プロジェクト全体で共通して利用する設定やパス、OpenAIクライアントを定義するためのものです。
# 各フェーズのコードから import される「共通ハブ」の役割を持ちます。

import os
from pathlib import Path
#「/」演算子を「パス結合用に上書き」する効果ももつ


from dotenv import load_dotenv#環境変数ファイル.envを読むためa
from openai import OpenAI #OpenAI APIを呼ぶための公式クライアント

# プロジェクトルート
#__file__：このファイル（config.py) 
# .resolve()：絶対パスに変換
#.parent：その1つ上のフォルダ
BASE_DIR = Path(__file__).resolve().parent

# """
# ゆくゆくは下記のようにフォルダをつなげたい
# BASE_DIR
#  └─ data
#      └─ processed

# だから
# 例えば、print(PROCESSED_DATA_DIR)
# →C:\Users\...\genai_expert_rag_project_v2\data\processed

# 変数BASE_DIRを作り、その上にいろいろフォルダを変数で組み立ててくイメージ
# だから、OS依存もしないし、v2のフォルダをごっそり移動させても正常に作動する
# """

#もし.envファイルがあれば、envファイルを読み込む
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY が設定されていません。.env を確認してください。")

# OpenAI クライアント（chat, embeddings で共通利用）
client = OpenAI(api_key=OPENAI_API_KEY)

# データディレクトリ
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDF_DIR = DATA_DIR / "pdf"
QA_PAIRS_DIR = DATA_DIR / "qa_pairs"

# """
# BASE_DIR (= プロジェクトの根っこ)
#  ├─ data
#  │   ├─ raw
#  │   └─ processed
#  └─ artifacts

# なぜこの書き方が強いか
# ・OS依存しない（Windows / Mac / Linux）
# ・v2フォルダを丸ごと移動しても壊れない
# ・絶対パスを1文字も書いていない

# 👉 実務・面接で評価される設計

# """


#フォルダ作成(なければ) （なければ作る。あってもエラーにならない)
for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PDF_DIR, QA_PAIRS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 成果物（インデックスや要約）を保存するディレクトリ
ARTIFACTS_DIR = BASE_DIR / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"
SUMMARY_DIR = ARTIFACTS_DIR / "summary"

for d in (INDEX_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)
