"""
转人工 Agent
====================
处理复杂/敏感问题，触发转人工

设计理由:
- 转人工是敏感兜底分支，处理投诉、情绪激动、复杂业务等场景
- 用LLM分析用户情绪并生成有温度的安抚性回复，比固定话术更自然
- 明确标记need_transfer=True，让LangGraph路由到人工坐席节点
- LLM失败时用预设安抚话术兜底，保证用户不被冷落
- 安抚回复遵循"共情+说明+告知转接"三段式结构
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


# 转人工安抚提示词
# 设计理由: 三段式结构(共情+说明+告知转接)让安抚更真诚，不敷衍
TRANSFER_SYSTEM_PROMPT = """你是XX商城的智能客服。用户的问题需要转交人工客服处理。

请根据用户消息生成一段安抚性回复，要求：
1. 先安抚用户情绪，表达理解和歉意
2. 简要说明为什么需要转人工（如：需要人工核实/涉及复杂处理）
3. 告知已为您转接人工客服，请稍候
4. 语气真诚、专业，不要敷衍
5. 用中文，2-4句话

【对话历史】
{history}"""


class TransferAgent:
    """处理复杂/敏感问题，触发转人工"""

    async def __call__(self, state: Dict) -> Dict:
        """
        转人工处理

        处理流程:
        1. 提取用户消息和对话历史
        2. LLM分析情绪并生成安抚回复(三段式:共情+说明+告知)
        3. 标记need_transfer=True供下游路由使用
        4. LLM失败时用预设安抚话术兜底

        Args:
            state: 包含 messages 字段，对话历史列表

        Returns:
            {"reply": str, "need_transfer": True}
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "reply": "您好，已为您转接人工客服，请稍候。",
                "need_transfer": True,
            }

        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 格式化最近5轮历史，让LLM理解上下文
        history = self._format_history(messages[:-1], max_turns=5)

        try:
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(
                    content=TRANSFER_SYSTEM_PROMPT.format(history=history)
                ),
                HumanMessage(content=user_text),
            ])
            reply = response.content.strip()
            logger.info(f"转人工安抚回复: {reply[:50]}...")
            return {"reply": reply, "need_transfer": True}
        except Exception as e:
            # LLM失败，用预设安抚话术兜底，保证转人工不中断
            logger.error(f"转人工LLM调用失败: {e}")
            return {
                "reply": (
                    "非常抱歉给您带来不好的体验，您的反馈我已记录。"
                    "已为您转接人工客服，请稍候片刻，会有专人为您处理。"
                ),
                "need_transfer": True,
            }

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
