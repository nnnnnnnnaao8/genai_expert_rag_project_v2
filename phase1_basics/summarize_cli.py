# このファイルは、PDFを読み込んでLLMに要約させるCLIツールを実装するためのものです。
# 「LLM APIを使ったシンプルなGen-AIシステム」の例として位置づけています。

import argparse

from config import client
from .etl import read_pdf_text

# ★開発用（対話型ウィンドウでも動かすためのおまじない）
try:
    import dev_bootstrap  # noqa: F401
except ImportError:
    pass

def summarize_text(text: str, max_tokens: int = 512) -> str:
    """
    本文テキストをLLMに渡して要約を生成する。
    """
    prompt = (
        "次の文章を日本語で要約してください。\n\n"
        f"{text[:4000]}\n\n"
        "### 要約:"
    )
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="PDF要約CLIツール")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="要約対象のPDFファイル名（data/pdf 配下）",
    )
    args = parser.parse_args()

    text = read_pdf_text(args.pdf)
    summary = summarize_text(text)
    print("=== 要約結果 ===")
    print(summary)


if __name__ == "__main__":
    main()
