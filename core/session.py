"""会话管理模块

基于内存的会话管理，维护对话历史和元数据。
当对话历史超过最大轮数时，自动压缩旧消息。
"""

from datetime import datetime

try:
    from config import settings
except ImportError:
    # 配置不可用时使用默认值
    class _FallbackSettings:
        MAX_HISTORY_TURNS = 10

    settings = _FallbackSettings()


class SessionManager:
    """会话管理器（内存版）

    维护每个session_id对应的对话历史和元数据。
    对话历史超过MAX_HISTORY_TURNS时自动压缩旧消息。
    """

    def __init__(self):
        self._sessions = {}  # session_id -> {"messages": [], "metadata": {}}
        self._max_turns = getattr(settings, "MAX_HISTORY_TURNS", 10)

    def _ensure_session(self, session_id: str) -> dict:
        """确保会话存在，不存在则创建"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "metadata": {
                    "created_at": None,
                    "last_active": None,
                    "turn_count": 0,
                },
            }
        return self._sessions[session_id]

    def get_history(self, session_id: str) -> list:
        """获取对话历史

        Args:
            session_id: 会话唯一标识

        Returns:
            对话历史列表，每项为 {"role": "user"/"assistant", "content": "..."}
        """
        session = self._ensure_session(session_id)
        return list(session["messages"])

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到对话历史

        当消息数量超过MAX_HISTORY_TURNS * 2（一轮=用户+助手）时，
        保留最近的消息，自动丢弃最早的记录。

        Args:
            session_id: 会话唯一标识
            role: 消息角色 ("user" / "assistant")
            content: 消息内容
        """
        session = self._ensure_session(session_id)
        now = datetime.now().isoformat()

        # 更新元数据
        meta = session["metadata"]
        if meta["created_at"] is None:
            meta["created_at"] = now
        meta["last_active"] = now
        meta["turn_count"] += 1

        # 添加消息
        session["messages"].append({"role": role, "content": content})

        # 压缩：超过最大轮数对应的消息数时，保留最近的消息
        # 一轮对话 = user + assistant = 2条消息
        max_messages = self._max_turns * 2
        if len(session["messages"]) > max_messages:
            session["messages"] = session["messages"][-max_messages:]

    def clear(self, session_id: str):
        """清空指定会话的历史记录

        Args:
            session_id: 会话唯一标识
        """
        if session_id in self._sessions:
            self._sessions[session_id]["messages"] = []
            self._sessions[session_id]["metadata"]["turn_count"] = 0

    def get_all_sessions(self) -> list:
        """获取所有会话摘要（管理面板用）

        Returns:
            会话摘要列表，每项含 session_id、消息数、创建时间、最后活跃时间等
        """
        sessions_summary = []
        for session_id, session in self._sessions.items():
            meta = session["metadata"]
            sessions_summary.append(
                {
                    "session_id": session_id,
                    "message_count": len(session["messages"]),
                    "turn_count": meta.get("turn_count", 0),
                    "created_at": meta.get("created_at"),
                    "last_active": meta.get("last_active"),
                }
            )
        return sessions_summary


# 全局会话管理实例
session_manager = SessionManager()
