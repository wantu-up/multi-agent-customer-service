"""
转人工 Agent
====================
处理复杂/敏感问题，以人工客服身份接管对话

设计理由:
- 转人工是敏感兜底分支，处理投诉、情绪激动、复杂业务等场景
- 第一次触发时用LLM生成有温度的安抚+接通回复
- 后续对话以人工客服身份直接回应用户问题，不再重复"正在转接"
- LLM失败时用预设话术兜底，保证用户不被冷落
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


# 人工客服系统提示词
# 设计理由: 以人工客服身份直接回应用户，语气专业、真诚、有温度
HUMAN_AGENT_PROMPT = """你是XX商城的人工客服代表，现在正在与用户直接对话。

请遵循以下原则：
1. 以人工客服身份回复，语气真诚、专业、有耐心
2. 直接回答用户的问题，不要说"转接"之类的话，你已经是人工客服了
3. 如果用户情绪激动，先安抚再解决问题
4. 如果问题超出你的能力范围，告知会反馈给专员处理
5. 用中文，回复简洁，2-4句话

【对话历史】
{history}"""


class TransferAgent:
    """处理复杂/敏感问题，以人工客服身份接管对话"""

    async def __call__(self, state: Dict) -> Dict:
        """
        转人工处理

        处理流程:
        1. 检查是否已经转接过（need_transfer已在state中）
        2. 首次转接：生成安抚+接通回复
        3. 后续对话：以人工客服身份直接回应用户
        4. 标记need_transfer=True
        5. LLM失败时用预设话术兜底

        Args:
            state: 包含 messages、need_transfer 字段

        Returns:
            {"reply": str, "need_transfer": True}
        """
        messages = state.get("messages", [])
        already_transferred = state.get("need_transfer", False)

        if not messages:
            return {
                "reply": "您好，我是人工客服，请问有什么可以帮您？",
                "need_transfer": True,
            }

        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 格式化最近5轮历史
        history = self._format_history(messages[:-1], max_turns=5)

        try:
            llm = get_llm()

            if not already_transferred:
                # 首次转接：安抚 + 接通
                response = await llm.ainvoke([
                    SystemMessage(content=(
                        "你是XX商城的人工客服。用户的问题需要人工处理，"
                        "请根据用户消息生成一段回复，要求：\n"
                        "1. 先安抚用户情绪，表达理解和歉意\n"
                        "2. 告知用户你已上线，可以直接为其处理问题\n"
                        "3. 语气真诚专业，2-3句话\n\n"
                        f"【对话历史】\n{history}"
                    )),
                    HumanMessage(content=user_text),
                ])
            else:
                # 后续对话：以人工客服身份直接回应
                response = await llm.ainvoke([
                    SystemMessage(
                        content=HUMAN_AGENT_PROMPT.format(history=history)
                    ),
                    HumanMessage(content=user_text),
                ])

            reply = response.content.strip()
            logger.info(f"人工客服回复: {reply[:50]}...")
            return {"reply": reply, "need_transfer": True}

        except Exception as e:
            logger.error(f"转人工LLM调用失败: {e}")
            if not already_transferred:
                return {
                    "reply": (
                        "您好，我是人工客服，很高兴为您服务。"
                        "请问您遇到了什么问题？我来帮您处理。"
                    ),
                    "need_transfer": True,
                }
            return {
                "reply": "您好，我在的，请问还有什么可以帮您的吗？",
                "need_transfer": True,
            }

    @staticmethod
    def _format_history(messages: list, max_turns: int = 5) -> str:
        """格式化对话历史，只取最近max_turns轮"""
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
