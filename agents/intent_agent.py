"""
意图识别 Agent
====================
识别用户意图，路由到对应子Agent

设计理由:
- 意图识别是多Agent客服系统的入口，决定了后续走哪个处理分支
- 用LLM做意图分类比纯规则匹配更灵活，能处理多变的用户表达
- 加入few-shot示例提升小模型(Qwen2.5-7B)的分类准确率
- 关键词硬规则作为兜底，保证转人工类敏感意图不漏判
- 返回confidence，低置信度时降级到闲聊，避免误路由到错误分支
"""

import json
import re
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


# 转人工硬规则关键词：命中即直接转人工，不依赖LLM判断
# 设计理由: 投诉/举报等敏感词若漏判会导致用户体验灾难，用硬规则兜底
TRANSFER_KEYWORDS = [
    # 投诉/情绪类
    "投诉", "差评", "退款失败", "举报",
    "工商", "315", "消费者协会", "律师",
    # 明确要求人工类
    "人工", "转人工", "人工客服", "真人", "真人客服",
    "找客服", "找人工", "接人工", "转接人工",
]

# 意图分类的系统提示词，包含4种意图定义和few-shot示例
# 设计理由: few-shot能显著提升7B小模型的分类稳定性
INTENT_SYSTEM_PROMPT = """你是一个用户意图分类器。请判断用户消息属于以下哪种意图，返回JSON。

## 意图定义
- kb_qa: 知识库问答。用户询问商品信息、售后政策、使用方法、常见问题等，需要检索知识库才能回答。
- tool_call: 工具调用。用户需要查询或操作订单，如查订单状态、查物流、申请退款等，需要调用业务工具。
- chitchat: 闲聊。打招呼、感谢、表情、与业务无关的日常对话。
- transfer: 转人工。用户明确要求人工客服、情绪激动、投诉、举报等。

## 示例
用户: "你们的退换货政策是什么？"
结果: {"intent": "kb_qa", "confidence": 0.9, "reason": "询问退换货政策，属于知识库问答"}

用户: "帮我查一下订单12345的物流到哪了"
结果: {"intent": "tool_call", "confidence": 0.95, "reason": "需要查询订单物流，调用工具"}

用户: "你好呀"
结果: {"intent": "chitchat", "confidence": 0.9, "reason": "打招呼，闲聊"}

用户: "我要投诉你们！服务太差了！"
结果: {"intent": "transfer", "confidence": 0.95, "reason": "用户投诉，情绪激动，需转人工"}

用户: "这个产品怎么使用？"
结果: {"intent": "kb_qa", "confidence": 0.85, "reason": "询问产品使用方法，知识库问答"}

用户: "订单888的退款进度怎么样了"
结果: {"intent": "tool_call", "confidence": 0.9, "reason": "查询退款进度，需调用工具"}

## 输出格式
只返回JSON，不要有任何其他内容：
{"intent": "kb_qa|tool_call|chitchat|transfer", "confidence": 0.0-1.0, "reason": "简短理由"}"""


class IntentAgent:
    """识别用户意图，路由到对应子Agent"""

    async def __call__(self, state: Dict) -> Dict:
        """
        识别用户意图

        处理流程:
        1. 提取用户最新消息
        2. 硬规则检查转人工关键词（优先级最高）
        3. LLM意图分类(few-shot)
        4. 解析JSON，低置信度降级到chitchat
        5. LLM失败时降级到chitchat，不中断流程

        Args:
            state: 包含 messages 字段，对话历史列表

        Returns:
            {"intent": str, "confidence": float, "reason": str}
        """
        messages = state.get("messages", [])
        if not messages:
            # 无消息时默认闲聊，保证流程不中断
            return {"intent": "chitchat", "confidence": 0.5, "reason": "无消息，默认闲聊"}

        # 获取最后一条用户消息
        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 步骤2: 硬规则检查转人工关键词（优先级最高，保证敏感意图不漏判）
        for kw in TRANSFER_KEYWORDS:
            if kw in user_text:
                logger.info(f"命中转人工关键词[{kw}]，直接路由到transfer")
                return {
                    "intent": "transfer",
                    "confidence": 1.0,
                    "reason": f"命中转人工关键词: {kw}",
                }

        # 步骤3: LLM意图分类
        try:
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=f'用户: "{user_text}"\n结果:'),
            ])
            raw = response.content.strip()

            # 步骤4: 解析JSON
            result = self._parse_json(raw)
            if result is None:
                logger.warning(f"意图JSON解析失败，降级chitchat。原始: {raw[:100]}")
                return {
                    "intent": "chitchat",
                    "confidence": 0.5,
                    "reason": "JSON解析失败，降级闲聊",
                }

            intent = result.get("intent", "chitchat")
            confidence = float(result.get("confidence", 0.5))
            reason = result.get("reason", "")

            # 意图校验：只接受4种合法意图，非法值降级
            valid_intents = {"kb_qa", "tool_call", "chitchat", "transfer"}
            if intent not in valid_intents:
                logger.warning(f"非法意图[{intent}]，降级chitchat")
                return {
                    "intent": "chitchat",
                    "confidence": 0.5,
                    "reason": f"非法意图: {intent}",
                }

            # 低置信度降级到闲聊，避免误路由到错误分支
            if confidence < 0.5:
                logger.info(f"低置信度[{confidence}]，降级chitchat")
                return {
                    "intent": "chitchat",
                    "confidence": confidence,
                    "reason": f"低置信度降级: {reason}",
                }

            logger.info(f"意图识别: {intent} (confidence={confidence})")
            return {"intent": intent, "confidence": confidence, "reason": reason}

        except Exception as e:
            # LLM调用失败，降级到闲聊（不能抛异常中断流程）
            logger.error(f"意图识别LLM调用失败: {e}，降级chitchat")
            return {
                "intent": "chitchat",
                "confidence": 0.0,
                "reason": f"LLM调用失败: {str(e)[:50]}",
            }

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        解析LLM返回的JSON

        兼容三种情况:
        1. 纯JSON文本
        2. ```json ... ``` 代码块包裹
        3. 文本中夹杂的 { ... } 对象

        解析失败返回None，由调用方做降级处理
        """
        # 先尝试提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        # 再尝试提取第一个 { ... } 对象
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
