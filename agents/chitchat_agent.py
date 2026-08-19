"""
闲聊 Agent
====================
处理非业务问题的通用对话

设计理由:
- 闲聊是兜底分支，处理打招呼、感谢、表情、与业务无关的日常对话
- 包含最近5轮对话历史，保持上下文连贯性
- 温度稍高增加回复自然度，但限制在客服角色范围内不越界
- LLM失败时用预设话术兜底，保证用户始终能得到响应
- 引导用户描述业务需求，自然过渡到可用工具/知识库
"""

import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings

logger = logging.getLogger(__name__)


def get_llm():
    """获取LLM实例（统一调用入口）"""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# 闲聊系统提示词
# 设计理由: 明确客服角色边界，闲聊中引导业务转化，不编造政策
CHITCHAT_SYSTEM_PROMPT = """你是XX商城的智能客服助手。当前用户在和你闲聊。

要求：
1. 友好、简洁、有温度，像朋友一样交流
2. 如果用户提到业务需求（查订单/退换货/投诉等），引导用户描述具体需求
3. 不要编造商城政策或做出承诺
4. 回复控制在2-3句话以内
5. 用中文回复

【对话历史】
{history}"""


class ChitchatAgent:
    """通用对话，处理非业务问题"""

    async def __call__(self, state: Dict) -> Dict:
        """
        闲聊回复

        处理流程:
        1. 提取用户最新消息和对话历史
        2. 构造系统提示(角色+最近5轮历史)
        3. LLM生成闲聊回复
        4. LLM失败时用预设话术兜底

        Args:
            state: 包含 messages 字段，对话历史列表

        Returns:
            {"reply": str}
        """
        messages = state.get("messages", [])
        if not messages:
            # 无消息时给开场白
            return {"reply": "您好，我是XX商城智能客服，请问有什么可以帮您？"}

        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 格式化最近5轮历史，保持上下文连贯
        history = self._format_history(messages[:-1], max_turns=5)

        try:
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(
                    content=CHITCHAT_SYSTEM_PROMPT.format(history=history)
                ),
                HumanMessage(content=user_text),
            ])
            reply = response.content.strip()
            logger.info(f"闲聊回复: {reply[:50]}...")
            return {"reply": reply}
        except Exception as e:
            # LLM失败，用预设话术兜底，不让对话中断
            logger.error(f"闲聊LLM调用失败: {e}")
            return {"reply": "您好，我在的，请问有什么可以帮您呢？"}

    @staticmethod
    def _format_history(messages: list, max_turns: int = 5) -> str:
        """
        格式化对话历史

        只取最近max_turns轮，避免上下文过长
        """
        recent = (
            messages[-max_turns * 2:]
            if len(messages) > max_turns * 2
            else messages
        )
        lines = []
        for msg in recent:
            if isinstance(msg, dict):
                role = "用户" if msg.get("role") == "user" else "客服"
                lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines) if lines else "（无历史记录）"
