# このファイルは、ユーザの質問内容に応じて「どのツールを呼ぶか」を決める簡易プランナーを実装するためのものです。
# if文レベルのシンプルなロジックですが、「AIエージェントの構造」を示すには十分な例になります。

from phase3_custom.agent.state import AgentState
from phase3_custom.agent.tools import tool_csv_summary, tool_rag_search


def classify_intent(query: str) -> str:
    """
    非LLMな超シンプル分類ロジック。
    将来的には LLM による意図分類に差し替えてもよい。
    """
    q = query.lower()
    if "売上" in q or "sales" in q or "集計" in q:
        return "csv_summary"
    else:
        return "rag_search"


def run_agent_once(state: AgentState, query: str) -> str:
    """
    エージェントを1ターン分だけ動かす。
    - ユーザ発話を状態に追加
    - 意図を分類
    - 対応するツールを呼び出し
    - 結果を状態に追加
    """
    state.add_user(query)
    intent = classify_intent(query)

    if intent == "csv_summary":
        result = tool_csv_summary()
    else:
        result = tool_rag_search(query)

    state.add_assistant(result)
    return result
