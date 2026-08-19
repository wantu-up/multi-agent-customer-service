"""
质检 Agent (QA)
====================
对AI回复做安全审核+满意度预估

设计理由:
- 质检是输出侧防线，防止不当回复触达用户，保护品牌声誉
- 用LLM做安全审核：检测辱骂、过度承诺、信息泄露、错误引导等风险
- 预估满意度作为质量信号，低分触发转人工，实现质量闭环
- safety=fail直接转人工，宁可多转不可漏放
- LLM失败时保守降级（标记需人工复核），宁严勿松
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


# 满意度阈值：低于此值触发转人工
SATISFACTION_THRESHOLD = 0.6


# 质检系统提示词
# 设计理由: 明确3个审核维度+评分标准，让LLM审核有据可依
QA_SYSTEM_PROMPT = """你是客服质检员。请对AI客服的回复做质量审核。

## 审核维度
1. 安全性(safety): 回复是否包含以下风险
   - pass: 无风险
   - warning: 轻微问题（如语气生硬、不够友好）
   - fail: 严重风险（辱骂用户、过度承诺、泄露隐私、错误引导）
2. 满意度(satisfaction): 0.0-1.0，预估用户满意程度
3. 问题(issues): 发现的具体问题列表

## 评估标准
- 回答了用户问题 +0.3
- 语气友好专业 +0.2
- 信息准确不过度承诺 +0.2
- 有实质帮助 +0.3

## 输出格式
只返回JSON，不要有任何其他内容：
{"safety": "pass|warning|fail", "satisfaction": 0.0-1.0, "issues": ["问题描述"]}"""


class QAAgent:
    """对回复做安全审核+满意度预估"""

    async def __call__(self, state: Dict) -> Dict:
        """
        质检审核

        处理流程:
        1. 提取用户消息和AI回复
        2. LLM审核安全性(safety)+满意度(satisfaction)
        3. 解析JSON结果
        4. safety=fail或满意度低于阈值 → 标记need_transfer
        5. LLM失败时保守降级（标记转人工，宁严勿松）

        Args:
            state: 包含 messages(用户消息) 和 reply(AI回复)

        Returns:
            {"risk_score": float, "satisfaction_score": float, "need_transfer": bool}
        """
        messages = state.get("messages", [])
        reply = state.get("reply", "")

        if not reply:
            # 无回复可审核，保守标记转人工
            logger.warning("质检时无AI回复，保守标记转人工")
            return {
                "risk_score": 0.5,
                "satisfaction_score": 0.0,
                "need_transfer": True,
            }

        # 获取用户最新消息
        user_text = ""
        if messages:
            last_msg = messages[-1]
            user_text = (
                last_msg.get("content", "")
                if isinstance(last_msg, dict)
                else str(last_msg)
            )

        try:
            llm = get_llm()
            response = await llm.ainvoke([
                SystemMessage(content=QA_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"用户消息: {user_text}\nAI回复: {reply}\n\n请审核:"
                ),
            ])
            raw = response.content.strip()
            result = self._parse_json(raw)

            if result is None:
                # JSON解析失败，保守降级
                logger.warning(f"质检JSON解析失败: {raw[:100]}")
                return self._fallback()

            safety = result.get("safety", "warning")
            satisfaction = float(result.get("satisfaction", 0.5))
            issues = result.get("issues", [])

            # 风险分数: safety=fail→高风险, warning→中风险, pass→低风险
            risk_map = {"fail": 0.9, "warning": 0.4, "pass": 0.1}
            risk_score = risk_map.get(safety, 0.4)

            # 转人工判定:
            # 1. safety=fail → 直接转人工（严重风险不放过）
            # 2. 满意度低于阈值 → 转人工（质量不达标）
            need_transfer = (safety == "fail") or (
                satisfaction < SATISFACTION_THRESHOLD
            )

            logger.info(
                f"质检完成: safety={safety}, satisfaction={satisfaction}, "
                f"transfer={need_transfer}, issues={issues}"
            )
            return {
                "risk_score": risk_score,
                "satisfaction_score": satisfaction,
                "need_transfer": need_transfer,
            }
        except Exception as e:
            # LLM失败，保守降级：标记需人工复核，宁严勿松
            logger.error(f"质检LLM调用失败: {e}")
            return self._fallback()

    @staticmethod
    def _fallback() -> Dict:
        """
        LLM失败时的保守降级

        设计理由: 质检失败时不知道回复是否有问题，
        保守策略是标记转人工让人工复核，宁可多转不可漏放
        """
        return {
            "risk_score": 0.5,
            "satisfaction_score": 0.5,
            "need_transfer": True,
        }

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
