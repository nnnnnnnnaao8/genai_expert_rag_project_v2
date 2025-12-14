# このファイルは、エージェントの状態（会話履歴など）を管理するためのクラスを定義しています。
# シンプルな dataclass ベースで、「状態を持つエージェント」のイメージを掴むことを目的にしています。

from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class AgentState:
    history: List[Turn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.history.append(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.history.append(Turn(role="assistant", content=content))

    def last_user_message(self) -> str:
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.content
        return ""
