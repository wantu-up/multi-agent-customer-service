"""LangGraph 多Agent编排路由模块

使用 LangGraph StateGraph 构建多Agent客服系统的流程编排：

    START
      |
      v
    intent_detection  (意图识别)
      |
      v (conditional_edge: route_by_intent)
      +--> kb_agent         (知识库问答)
      +--> tool_agent       (工具调用)
      +--> chitchat_agent   (闲聊)
      +--> transfer_agent   (转人工)
      |
      v
    qa_agent  (质检评分)
      |
      v
    END

agents包尚未创建时，通过try/except优雅降级，
确保router模块可独立import和运行。
"""

import time
from datetime import datetime

from langgraph.graph import StateGraph, END

from core.state import AgentState
from core.session import session_manager


# ====================================================================
# 优雅导入各Agent（agents包可能尚未创建，import失败时降级为None）
# ====================================================================

_INTENT_AGENT = None
_KB_AGENT = None
_TOOL_AGENT = None
_CHITCHAT_AGENT = None
_TRANSFER_AGENT = None
_QA_AGENT = None

try:
    from agents.intent_agent import IntentAgent
    _INTENT_AGENT = IntentAgent()
except ImportError:
    pass

try:
    from agents.kb_agent import KBAgent
    _KB_AGENT = KBAgent()
except ImportError:
    pass

try:
    from agents.tool_agent import ToolAgent
    _TOOL_AGENT = ToolAgent()
except ImportError:
    pass

try:
    from agents.chitchat_agent import ChitchatAgent
    _CHITCHAT_AGENT = ChitchatAgent()
except ImportError:
    pass

try:
    from agents.transfer_agent import TransferAgent
    _TRANSFER_AGENT = TransferAgent()
except ImportError:
    pass

try:
    from agents.qa_agent import QAAgent
    _QA_AGENT = QAAgent()
except ImportError:
    pass


# ====================================================================
# 路由函数
# ====================================================================

def route_by_intent(state: AgentState) -> str:
    """路由函数：根据意图返回目标节点名

    Args:
        state: 当前AgentState

    Returns:
        目标节点名称字符串
    """
    intent = state.get("intent", "chitchat")
    mapping = {
        "kb_qa": "kb_agent",
        "tool_call": "tool_agent",
        "chitchat": "chitchat_agent",
        "transfer": "transfer_agent",
    }
    return mapping.get(intent, "chitchat_agent")


# ====================================================================
# 多Agent路由编排器
# ====================================================================

class MultiAgentRouter:
    """多Agent路由编排器

    使用LangGraph StateGraph构建多Agent客服流程：
    意图识别 -> 条件路由 -> 各专业Agent -> QA质检 -> 结束

    agents包不存在时自动降级为基础规则回复。
    """

    def __init__(self):
        self._compiled_graph = None
        self._build_graph()

    # ----------------------------------------------------------------
    # 图构建
    # ----------------------------------------------------------------

    def _build_graph(self):
        """构建LangGraph状态机"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("kb_agent", self._kb_agent_node)
        workflow.add_node("tool_agent", self._tool_agent_node)
        workflow.add_node("chitchat_agent", self._chitchat_agent_node)
        workflow.add_node("transfer_agent", self._transfer_agent_node)
        workflow.add_node("qa_agent", self._qa_agent_node)

        # 设置入口节点
        workflow.set_entry_point("intent_detection")

        # 条件路由：intent_detection 根据意图分流到各业务Agent
        workflow.add_conditional_edges(
            "intent_detection",
            route_by_intent,
            {
                "kb_agent": "kb_agent",
                "tool_agent": "tool_agent",
                "chitchat_agent": "chitchat_agent",
                "transfer_agent": "transfer_agent",
            },
        )

        # 各业务Agent执行后统一进入qa_agent质检节点
        workflow.add_edge("kb_agent", "qa_agent")
        workflow.add_edge("tool_agent", "qa_agent")
        workflow.add_edge("chitchat_agent", "qa_agent")
        workflow.add_edge("transfer_agent", "qa_agent")

        # qa_agent -> END
        workflow.add_edge("qa_agent", END)

        # 编译图
        self._compiled_graph = workflow.compile()

    # ----------------------------------------------------------------
    # 节点函数（均为async，接收AgentState返回partial AgentState）
    # ----------------------------------------------------------------

    async def _intent_detection_node(self, state: AgentState) -> dict:
        """意图识别节点"""
        if _INTENT_AGENT is not None:
            try:
                result = await _INTENT_AGENT.detect(state)
                return result
            except Exception:
                pass
        # 降级：基于关键词规则
        return self._fallback_intent_detection(state)

    async def _kb_agent_node(self, state: AgentState) -> dict:
        """知识库问答Agent节点"""
        if _KB_AGENT is not None:
            try:
                result = await _KB_AGENT.run(state)
                return result
            except Exception:
                pass
        # 降级：直接调用知识库检索
        from core.knowledge_base import knowledge_base

        last_msg = state["messages"][-1]["content"] if state.get("messages") else ""
        context = knowledge_base.get_context(last_msg)
        return {
            "kb_context": context,
            "reply": (
                "（知识库Agent降级模式）根据您的问题，为您检索到以下信息：\n\n"
                + context
            ) if context else "抱歉，未检索到相关问题的解答，建议您联系人工客服获取帮助。",
        }

    async def _tool_agent_node(self, state: AgentState) -> dict:
        """工具调用Agent节点"""
        if _TOOL_AGENT is not None:
            try:
                result = await _TOOL_AGENT.run(state)
                return result
            except Exception:
                pass
        # 降级：提示转人工
        return {
            "tool_result": "（工具Agent降级模式）工具调用功能暂不可用。",
            "need_transfer": True,
            "reply": "您的订单/物流查询需求已收到，工具服务正在维护中，正在为您转接人工客服，请稍候。",
        }

    async def _chitchat_agent_node(self, state: AgentState) -> dict:
        """闲聊Agent节点"""
        if _CHITCHAT_AGENT is not None:
            try:
                result = await _CHITCHAT_AGENT.run(state)
                return result
            except Exception:
                pass
        # 降级：固定回复
        return {
            "reply": "您好！很高兴为您服务，请问有什么可以帮您的吗？您可以咨询订单查询、物流跟踪、退换货政策等问题。",
        }

    async def _transfer_agent_node(self, state: AgentState) -> dict:
        """转人工Agent节点"""
        if _TRANSFER_AGENT is not None:
            try:
                result = await _TRANSFER_AGENT.run(state)
                return result
            except Exception:
                pass
        # 降级：固定转人工回复
        return {
            "need_transfer": True,
            "reply": "您的需求已记录，正在为您转接人工客服，请稍候。人工客服将在1-2分钟内与您联系，感谢您的耐心等待。",
        }

    async def _qa_agent_node(self, state: AgentState) -> dict:
        """质检Agent节点（风险评估 + 满意度预估）"""
        if _QA_AGENT is not None:
            try:
                result = await _QA_AGENT.run(state)
                return result
            except Exception:
                pass
        # 降级：基于关键词的风险评分
        return self._fallback_qa(state)

    # ----------------------------------------------------------------
    # 降级方法
    # ----------------------------------------------------------------

    def _fallback_intent_detection(self, state: AgentState) -> dict:
        """降级意图识别：基于关键词规则

        风险关键词 -> transfer
        工具关键词 -> tool_call
        默认 -> chitchat
        """
        try:
            from config import settings

            risk_keywords = settings.RISK_KEYWORDS
        except ImportError:
            risk_keywords = ["投诉", "差评", "退款", "举报", "律师"]

        last_msg = ""
        if state.get("messages"):
            last_msg = state["messages"][-1].get("content", "")

        # 1. 风险关键词 -> 转人工
        for kw in risk_keywords:
            if kw in last_msg:
                return {
                    "intent": "transfer",
                    "intent_confidence": 0.8,
                    "metadata": {"intent_method": "keyword_fallback"},
                }

        # 2. 工具相关关键词 -> 工具调用
        tool_keywords = ["订单", "物流", "快递", "退款", "发货", "运单", "到货"]
        for kw in tool_keywords:
            if kw in last_msg:
                return {
                    "intent": "tool_call",
                    "intent_confidence": 0.7,
                    "metadata": {"intent_method": "keyword_fallback"},
                }

        # 3. 知识库相关关键词
        kb_keywords = ["退货", "政策", "支付", "积分", "优惠券", "运费", "保修",
                        "密码", "发票", "尺码", "换货", "会员"]
        for kw in kb_keywords:
            if kw in last_msg:
                return {
                    "intent": "kb_qa",
                    "intent_confidence": 0.65,
                    "metadata": {"intent_method": "keyword_fallback"},
                }

        # 4. 默认闲聊
        return {
            "intent": "chitchat",
            "intent_confidence": 0.5,
            "metadata": {"intent_method": "keyword_fallback"},
        }

    def _fallback_qa(self, state: AgentState) -> dict:
        """降级质检：基于关键词的风险评分与满意度预估"""
        try:
            from config import settings

            risk_keywords = settings.RISK_KEYWORDS
            threshold = settings.SATISFACTION_THRESHOLD
        except ImportError:
            risk_keywords = ["投诉", "差评", "退款", "举报", "律师"]
            threshold = 0.6

        # 检查回复和用户消息中的风险关键词
        reply = state.get("reply", "")
        messages = state.get("messages", [])
        all_text = reply
        if messages:
            all_text += " " + messages[-1].get("content", "")

        risk_score = 0.0
        for kw in risk_keywords:
            if kw in all_text:
                risk_score += 0.2
        risk_score = min(risk_score, 1.0)

        satisfaction = max(0.0, 1.0 - risk_score)
        need_transfer = satisfaction < threshold or state.get("need_transfer", False)

        return {
            "risk_score": round(risk_score, 2),
            "satisfaction_score": round(satisfaction, 2),
            "need_transfer": need_transfer,
        }

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------

    def compile(self):
        """返回编译后的graph"""
        return self._compiled_graph

    async def run(self, session_id: str, user_message: str) -> dict:
        """执行完整多Agent流程

        Args:
            session_id: 会话ID
            user_message: 用户输入消息

        Returns:
            最终状态字典，包含 reply、intent、risk_score、satisfaction_score 等
        """
        start_time = time.time()

        # 记录用户消息到会话历史
        session_manager.add_message(session_id, "user", user_message)

        # 获取对话历史
        history = session_manager.get_history(session_id)

        # 构建初始状态
        initial_state: AgentState = {
            "messages": history,
            "session_id": session_id,
            "intent": "",
            "intent_confidence": 0.0,
            "tool_result": "",
            "kb_context": "",
            "reply": "",
            "risk_score": 0.0,
            "satisfaction_score": 0.0,
            "need_transfer": False,
            "metadata": {
                "route_history": [],
                "start_time": datetime.now().isoformat(),
            },
        }

        # 执行graph（异步）
        final_state = await self._compiled_graph.ainvoke(initial_state)

        # 记录耗时和结束时间
        elapsed = round(time.time() - start_time, 3)
        if "metadata" not in final_state:
            final_state["metadata"] = {}
        final_state["metadata"]["elapsed_seconds"] = elapsed
        final_state["metadata"]["end_time"] = datetime.now().isoformat()

        # 记录助手回复到会话历史
        reply = final_state.get("reply", "")
        if reply:
            session_manager.add_message(session_id, "assistant", reply)

        return final_state


# 全局路由实例
multi_agent_router = MultiAgentRouter()
