# このファイルは、RAG＋簡易エージェントをブラウザ上で試せるStreamlitアプリを提供するためのものです。
# 必須ではありませんが、PoCデモの見栄えを良くする用途を想定しています。

import streamlit as st

from phase3_custom.agent.state import AgentState
from phase3_custom.agent.planner import run_agent_once


def main():
    st.title("GenAI Expert RAG Demo (V2)")

    if "agent_state" not in st.session_state:
        st.session_state["agent_state"] = AgentState()

    query = st.text_input("質問を入力してください:")

    if st.button("送信") and query:
        state = st.session_state["agent_state"]
        result = run_agent_once(state, query)
        st.write("### 回答")
        st.write(result)

        with st.expander("会話履歴"):
            for turn in state.history:
                role = "ユーザ" if turn.role == "user" else "AI"
                st.markdown(f"**{role}:** {turn.content}")


if __name__ == "__main__":
    main()
