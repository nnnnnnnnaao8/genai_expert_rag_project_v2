# このファイルは、プロジェクト全体で共通して利用する設定やパス、OpenAIクライアントを定義するためのものです。
# 各フェーズのコードから import される「共通ハブ」の役割を持ちます。

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# プロジェクトルート
BASE_DIR = Path(__file__).resolve().parent

# .env を読み込む
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

for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PDF_DIR, QA_PAIRS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 成果物（インデックスや要約）を保存するディレクトリ
ARTIFACTS_DIR = BASE_DIR / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"
SUMMARY_DIR = ARTIFACTS_DIR / "summary"

for d in (INDEX_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)
