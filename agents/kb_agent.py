"""
知识库问答 Agent
====================
RAG检索增强生成，回答商品/售后/政策类问题

设计理由:
- RAG比纯LLM回答更准确，能引用知识库的具体内容而非编造
- 检索+生成两阶段：先检索相关文档片段，再基于上下文生成回复
- 限制LLM只能基于检索到的上下文回答，减少幻觉
- 检索失败或无结果时降级到通用回复，不能让用户等不到响应
- LLM生成失败时直接返回检索到的原文摘要作为兜底
"""

import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from core.knowledge_base import KnowledgeBase

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


# RAG生成的系统提示词
# 设计理由: 明确要求"只基于知识库回答"，防止LLM脱离上下文编造信息
KB_SYSTEM_PROMPT = """你是XX商城的智能客服。请根据以下知识库上下文回答用户问题。

要求：
1. 只基于下方【知识库】内容回答，不要编造信息
2. 如果知识库中没有相关信息，诚实告知并建议联系人工客服
3. 回答简洁、专业、友好
4. 涉及具体政策/数字时，严格引用知识库原文

【知识库】
{context}

【对话历史】
{history}"""


class KBAgent:
    """RAG检索+生成回复"""

    def __init__(self):
        # 知识库实例，负责向量检索
        self.kb = KnowledgeBase()

    async def __call__(self, state: Dict) -> Dict:
        """
        知识库问答

        处理流程:
        1. 提取用户最新消息作为检索query
        2. 调用kb.get_context检索相关知识片段
        3. 构造RAG prompt(系统+上下文+历史+问题)
        4. LLM基于上下文生成回复
        5. 返回回复和检索到的上下文

        Args:
            state: 包含 messages 字段，对话历史列表

        Returns:
            {"reply": str, "kb_context": str}
        """
        messages = state.get("messages", [])
        if not messages:
            return {"reply": "您好，请问有什么可以帮您？", "kb_context": ""}

        # 获取最后一条用户消息作为检索query
        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 步骤2: 检索知识库上下文
        try:
            context = self.kb.get_context(user_text)
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            context = ""

        # 检索不到上下文时降级回复，避免无依据生成
        if not context or not context.strip():
            logger.info("知识库无相关内容，降级回复")
            return {
                "reply": (
                    "抱歉，关于这个问题我暂时没有相关信息，"
                    "建议您联系人工客服获取准确答复。"
                ),
                "kb_context": "",
            }

        # 步骤3: 构造对话历史（最近5轮，避免prompt过长）
        history = self._format_history(messages[:-1], max_turns=5)

        # 步骤4: LLM基于上下文生成回复
        try:
            llm = get_llm()
            system_content = KB_SYSTEM_PROMPT.format(
                context=context, history=history
            )
            response = await llm.ainvoke([
                SystemMessage(content=system_content),
                HumanMessage(content=user_text),
            ])
            reply = response.content.strip()
            logger.info(f"知识库问答生成: {reply[:50]}...")
            return {"reply": reply, "kb_context": context}
        except Exception as e:
            # LLM失败降级：返回检索到的上下文摘要，不让用户空手而归
            logger.error(f"知识库问答LLM调用失败: {e}")
            return {
                "reply": (
                    f"根据我们的知识库，相关信息如下：\n"
                    f"{context[:200]}\n\n"
                    f"如需更多帮助请联系人工客服。"
                ),
                "kb_context": context,
            }

    @staticmethod
    def _format_history(messages: list, max_turns: int = 5) -> str:
        """
        格式化对话历史

        只取最近max_turns轮，避免上下文过长导致token超限
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
