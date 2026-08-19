"""LangGraph 状态定义模块

定义多Agent客服系统中各个节点之间共享的状态结构。
"""

from typing import TypedDict


class AgentState(TypedDict):
    """多Agent客服系统的全局状态结构

    在整个LangGraph流程中传递，每个节点读取所需字段并更新产出字段。
    """

    messages: list  # 对话历史 [{"role": "user"/"assistant", "content": "..."}]
    session_id: str  # 会话唯一标识
    intent: str  # 意图标签: "kb_qa" / "tool_call" / "chitchat" / "transfer"
    intent_confidence: float  # 意图识别置信度 0~1
    tool_result: str  # 工具调用结果
    kb_context: str  # RAG检索到的上下文
    reply: str  # 最终回复
    risk_score: float  # 质检风险分 0~1
    satisfaction_score: float  # 预估满意度 0~1
    need_transfer: bool  # 是否需要转人工
    metadata: dict  # 其他元数据(路由历史、耗时等)
