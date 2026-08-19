"""
工具调用 Agent
====================
根据用户需求调用业务工具(查订单/查物流/退款)

设计理由:
- 工具调用分两步：先让LLM决策调哪个工具+提取参数，再执行
- 不用function calling原生能力，而是用JSON输出+手工路由，更可控、更易调试
- 工具执行结果交给LLM做自然语言润色，避免直接返回原始JSON给用户
- LLM输出不合法JSON或工具不存在时降级，不能让流程中断
- 参数缺失时引导用户补充，而非盲目调用
"""

import json
import re
import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from core.tools import CustomerTools

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


# 工具决策提示词：让LLM判断调哪个工具+提取参数
# 设计理由: 用few-shot示例让7B小模型稳定输出结构化JSON
TOOL_DECISION_PROMPT = """你是客服系统的工具路由器。根据用户消息判断需要调用哪个工具，并提取参数。

## 可用工具
1. check_order: 查询订单状态。参数: {"order_id": "订单号"}
2. check_logistics: 查询物流信息。参数: {"order_id": "订单号"}
3. request_refund: 申请退款。参数: {"order_id": "订单号", "reason": "退款原因"}

## 示例
用户: "查一下订单12345的状态"
结果: {"tool": "check_order", "params": {"order_id": "12345"}}

用户: "订单88888到哪了，物流怎样"
结果: {"tool": "check_logistics", "params": {"order_id": "88888"}}

用户: "订单999我要退款，商品有质量问题"
结果: {"tool": "request_refund", "params": {"order_id": "999", "reason": "商品有质量问题"}}

用户: "帮我看看单号ABC123的快递"
结果: {"tool": "check_logistics", "params": {"order_id": "ABC123"}}

## 输出格式
只返回JSON，不要有任何其他内容：
{"tool": "check_order|check_logistics|request_refund", "params": {...}}
如果用户没有提供订单号，params中order_id设为空字符串""。"""


# 回复生成提示词：用工具结果生成自然语言回复
REPLY_GENERATION_PROMPT = """你是XX商城的智能客服。根据工具查询结果，用自然语言回复用户。

要求：
1. 用友好、简洁的语气转述查询结果
2. 不要暴露内部工具名/JSON结构
3. 如果查询失败或无结果，诚实告知并建议联系人工客服
4. 用中文回复

用户问题: {question}
查询结果: {tool_result}"""


class ToolAgent:
    """根据用户需求调用工具(查订单/查物流/退款)"""

    def __init__(self):
        # 业务工具实例，负责实际查询/操作
        self.tools = CustomerTools()

    async def __call__(self, state: Dict) -> Dict:
        """
        工具调用

        处理流程:
        1. LLM决策: 判断调哪个工具+提取参数(输出JSON)
        2. 校验工具名和参数合法性
        3. 执行工具调用获取结果
        4. LLM用工具结果生成自然语言回复
        5. 返回回复和工具结果

        Args:
            state: 包含 messages 字段，对话历史列表

        Returns:
            {"reply": str, "tool_result": str}
        """
        messages = state.get("messages", [])
        if not messages:
            return {
                "reply": "您好，请问需要查询什么订单信息？",
                "tool_result": "",
            }

        last_msg = messages[-1]
        user_text = (
            last_msg.get("content", "")
            if isinstance(last_msg, dict)
            else str(last_msg)
        )

        # 步骤1: LLM决策调用哪个工具
        try:
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(content=TOOL_DECISION_PROMPT),
                HumanMessage(content=f'用户: "{user_text}"\n结果:'),
            ])
            raw = response.content.strip()
            decision = self._parse_json(raw)
        except Exception as e:
            logger.error(f"工具决策LLM调用失败: {e}")
            return {
                "reply": "抱歉，系统暂时无法处理您的请求，请稍后再试或联系人工客服。",
                "tool_result": "",
            }

        # JSON解析失败降级：引导用户补充信息
        if decision is None:
            logger.warning(f"工具决策JSON解析失败: {raw[:100]}")
            return {
                "reply": (
                    "我理解您需要查询订单相关信息，"
                    "请提供您的订单号，我来为您查询。"
                ),
                "tool_result": "",
            }

        tool_name = decision.get("tool", "")
        params = decision.get("params", {})

        # 步骤2: 校验工具名合法性
        valid_tools = {"check_order", "check_logistics", "request_refund"}
        if tool_name not in valid_tools:
            logger.warning(f"非法工具名[{tool_name}]，降级")
            return {
                "reply": (
                    "抱歉，暂时无法识别您的具体需求，"
                    "请描述您想查询的订单信息。"
                ),
                "tool_result": "",
            }

        # 校验订单号是否存在
        order_id = params.get("order_id", "")
        if not order_id:
            logger.info(f"工具[{tool_name}]缺少订单号，引导用户补充")
            return {
                "reply": "请您提供订单号，我来为您查询相关信息。",
                "tool_result": "",
            }

        # 步骤3: 执行工具调用
        try:
            tool_result = await self._execute_tool(tool_name, params)
            logger.info(
                f"工具[{tool_name}]执行完成: {str(tool_result)[:80]}"
            )
        except Exception as e:
            logger.error(f"工具[{tool_name}]执行失败: {e}")
            tool_result = {"error": f"工具执行失败: {str(e)[:50]}"}

        # 步骤4: LLM用工具结果生成自然语言回复
        try:
            llm = get_llm()
            reply_response = await llm.ainvoke([
                SystemMessage(content=REPLY_GENERATION_PROMPT.format(
                    question=user_text,
                    tool_result=str(tool_result),
                )),
                HumanMessage(content="请生成回复"),
            ])
            reply = reply_response.content.strip()
        except Exception as e:
            # LLM失败，降级为直接返回工具结果摘要
            logger.error(f"回复生成LLM失败: {e}")
            reply = f"查询结果: {str(tool_result)[:200]}"

        return {"reply": reply, "tool_result": str(tool_result)}

    async def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """
        根据工具名执行对应方法

        将LLM决策的工具名映射到CustomerTools的具体方法
        """
        if tool_name == "check_order":
            return await self.tools.check_order(params.get("order_id", ""))
        elif tool_name == "check_logistics":
            return await self.tools.check_logistics(
                params.get("order_id", "")
            )
        elif tool_name == "request_refund":
            return await self.tools.request_refund(
                params.get("order_id", ""),
                params.get("reason", ""),
            )
        else:
            return {"error": f"未知工具: {tool_name}"}

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        解析LLM返回的JSON

        兼容纯JSON/代码块包裹/文本夹杂等情况
        解析失败返回None，由调用方做降级处理
        """
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
