# -*- coding: utf-8 -*-
"""
多 Agent 智能客服系统 - FastAPI 主程序
======================================
后端服务入口,提供 Web 界面与 REST API。

设计原则: 延迟导入 + 优雅降级
- 核心模块 (config / core.router / core.session) 全部用 try/except 包裹
- core.router 不可用时, /api/chat 降级为关键词路由 + 质检,返回降级标记
- 健康检查 /api/health 不依赖 router,可独立反映服务状态
- 静态文件挂载 static 目录

说明: 真实 core.router 基于 LangGraph 编排,其 run() 内部已向全局
session_manager 写入对话历史。本程序使用 *独立* 的 SessionManager 实例
作为管理面板/历史接口的数据源,二者互不干扰,不会产生重复消息。

启动: uvicorn main:app --host 0.0.0.0 --port 8002
"""

import os
import time
import uuid
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==================== 优雅降级: 配置模块 ====================
# config 为轻量级模块,缺失时使用环境变量兜底,保证服务可启动
try:
    from config import settings  # type: ignore
    _CONFIG_OK = True
    _CONFIG_ERR = ""
except Exception as _e:  # noqa: BLE001
    _CONFIG_OK = False
    _CONFIG_ERR = str(_e)

    class _FallbackSettings:
        """降级配置: 从环境变量读取,保证服务可启动"""
        APP_NAME: str = "多Agent智能客服系统"
        APP_VERSION: str = "1.0.0"
        LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("SILICONFLOW_API_KEY", ""))
        LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "8002"))
        DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
        SATISFACTION_THRESHOLD: float = float(os.getenv("SATISFACTION_THRESHOLD", "0.6"))

    settings = _FallbackSettings()  # type: ignore


# ==================== 优雅降级: 会话管理 ====================
# core.session 缺失时,提供兼容版会话管理器,接口与真实 SessionManager 对齐:
#   add_message(session_id, role, content) / get_history(session_id)
#   clear(session_id) / get_all_sessions()
try:
    from core.session import SessionManager  # type: ignore
    _SESSION_OK = True
    _SESSION_ERR = ""
except Exception as _e:  # noqa: BLE001
    _SESSION_OK = False
    _SESSION_ERR = str(_e)

    class SessionManager:  # type: ignore[no-redef]
        """降级版会话管理器 (纯内存,接口对齐真实 SessionManager)"""

        def __init__(self):
            self._sessions: dict = {}

        def _ensure(self, session_id: str) -> dict:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "messages": [],
                    "metadata": {"created_at": None, "last_active": None, "turn_count": 0},
                }
            return self._sessions[session_id]

        def add_message(self, session_id: str, role: str, content: str):
            s = self._ensure(session_id)
            now = datetime.now().isoformat()
            meta = s["metadata"]
            if meta["created_at"] is None:
                meta["created_at"] = now
            meta["last_active"] = now
            meta["turn_count"] += 1
            s["messages"].append({"role": role, "content": content, "timestamp": now})

        def get_history(self, session_id: str) -> list:
            s = self._ensure(session_id)
            return list(s["messages"])

        def clear(self, session_id: str):
            if session_id in self._sessions:
                self._sessions[session_id]["messages"] = []
                self._sessions[session_id]["metadata"]["turn_count"] = 0

        def get_all_sessions(self) -> list:
            out = []
            for sid, s in self._sessions.items():
                m = s["metadata"]
                out.append({
                    "session_id": sid,
                    "message_count": len(s["messages"]),
                    "turn_count": m.get("turn_count", 0),
                    "created_at": m.get("created_at"),
                    "last_active": m.get("last_active"),
                })
            out.sort(key=lambda x: x.get("last_active") or "", reverse=True)
            return out


# ==================== 优雅降级: 多 Agent 路由 ====================
# core.router 缺失时, /api/chat 走降级路由(关键词意图识别 + 质检)
try:
    from core.router import MultiAgentRouter  # type: ignore
    _ROUTER_OK = True
    _ROUTER_ERR = ""
except Exception as _e:  # noqa: BLE001
    _ROUTER_OK = False
    _ROUTER_ERR = str(_e)
    MultiAgentRouter = None  # type: ignore


# ==================== 降级路由: 关键词意图识别 + 质检 ====================
# 当 core.router 不可用时, 用关键词匹配模拟"意图识别→子Agent→质检"流程
_KB_QA_KEYWORDS = ["怎么", "如何", "什么是", "介绍", "功能", "价格", "多少钱", "使用", "区别", "能否", "可以吗", "支持", "政策", "保修"]
_ORDER_KEYWORDS = ["订单", "物流", "发货", "快递", "到哪", "什么时候到", "运单", "配送", "签收"]
_REFUND_KEYWORDS = ["退款", "退货", "换货", "售后", "坏了", "破损", "质量问题", "赔偿"]
_TRANSFER_KEYWORDS = ["人工", "真人", "转人工", "找客服", "人工服务", "坐席"]
_COMPLAINT_KEYWORDS = ["投诉", "举报", "差评", "不满", "态度", "律师"]


def _degraded_route(message: str) -> dict:
    """降级版多 Agent 流程: 意图识别→路由→子Agent→质检(关键词匹配)"""

    if any(k in message for k in _TRANSFER_KEYWORDS):
        intent, confidence, agent = "transfer", 0.78, "转人工Agent"
        reply = ("已为您转接人工客服,请稍候。"
                 "您也可以直接描述问题,我会在转接前为您做初步记录。")
        need_transfer = True
    elif any(k in message for k in _COMPLAINT_KEYWORDS):
        intent, confidence, agent = "complaint", 0.72, "投诉处理Agent"
        reply = ("非常抱歉给您带来不愉快的体验,您的投诉已记录并升级处理,"
                 "客服主管将在2小时内与您联系。")
        need_transfer = True
    elif any(k in message for k in _REFUND_KEYWORDS):
        intent, confidence, agent = "refund", 0.82, "售后退款Agent"
        reply = ("很抱歉给您带来不便,退款/售后申请已受理,"
                 "售后专员将在24小时内为您处理,请保留订单号便于跟进。")
        need_transfer = False
    elif any(k in message for k in _ORDER_KEYWORDS):
        intent, confidence, agent = "order_query", 0.85, "订单查询Agent"
        reply = ("已为您查询订单状态:您的订单正在配送中,"
                 "预计1-2个工作日内送达。如需查看物流详情请提供订单号。")
        need_transfer = False
    elif any(k in message for k in _KB_QA_KEYWORDS):
        intent, confidence, agent = "kb_qa", 0.88, "知识库问答Agent"
        reply = ("根据知识库:这是降级模式的示例回复,多Agent核心模块(LLM路由)"
                 "未加载,当前以关键词匹配代替意图识别。完整版将调用知识库检索+LLM生成。")
        need_transfer = False
    else:
        intent, confidence, agent = "general", 0.50, "通用闲聊Agent"
        reply = ("您好,我是多Agent智能客服助手。当前为降级模式,"
                 "可帮您处理产品咨询、订单查询、退款售后等问题,"
                 "也可以说“转人工”接入坐席。")
        need_transfer = False

    satisfaction = round(0.65 + confidence * 0.25, 3)
    if need_transfer:
        satisfaction = round(satisfaction * 0.9, 3)

    return {
        "reply": reply,
        "intent": intent,
        "intent_confidence": confidence,
        "satisfaction_score": satisfaction,
        "need_transfer": need_transfer,
        "metadata": {
            "route": _build_route_label(intent, degraded=True),
            "degraded": True,
            "agent": agent,
            "agents_invoked": ["intent_recognition", agent, "quality_check"],
            "fallback_reason": _ROUTER_ERR or "core.router not loaded",
        },
    }


# ==================== 路由标签归一化 ====================
# 真实 router 的 metadata 不含 "route" 字符串,这里按 intent 统一构造可读路由
_AGENT_BY_INTENT = {
    "kb_qa": "知识库问答Agent",
    "tool_call": "工具调用Agent",
    "chitchat": "闲聊Agent",
    "transfer": "转人工Agent",
    "order_query": "订单查询Agent",
    "refund": "售后退款Agent",
    "complaint": "投诉处理Agent",
    "general": "通用闲聊Agent",
}


def _build_route_label(intent: str, degraded: bool = False) -> str:
    agent = _AGENT_BY_INTENT.get(intent or "", "通用Agent")
    return "意图识别→" + agent + "→质检"


# ==================== 日志配置 ====================
# 真实 config.Settings 可能不含 LOG_LEVEL, 用 getattr 兜底
_log_level_name = getattr(settings, "LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, _log_level_name, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("multi_agent_cs")

if not _ROUTER_OK:
    logger.warning("core.router 加载失败, /api/chat 将降级为关键词路由: %s", _ROUTER_ERR)
if not _SESSION_OK:
    logger.warning("core.session 加载失败, 使用内存版会话管理器: %s", _SESSION_ERR)
if not _CONFIG_OK:
    logger.warning("config 模块加载失败, 使用环境变量降级配置: %s", _CONFIG_ERR)


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title=getattr(settings, "APP_NAME", "多Agent智能客服系统"),
    version=getattr(settings, "APP_VERSION", "1.0.0"),
    description="基于意图识别+多Agent路由+质检的智能客服系统",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 挂载静态文件目录 (挂载后 /static/xxx 可访问; 页面路由单独处理)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================== 组件实例 ====================
# router 可能为 None (降级); 实例化也用 try/except, 防止图构建失败导致启动崩溃
router = None
if MultiAgentRouter is not None:
    try:
        router = MultiAgentRouter()
    except Exception as e:  # noqa: BLE001
        logger.warning("MultiAgentRouter 实例化失败, 降级为关键词路由: %s", e)
        router = None

# 独立的会话管理实例 (不与 router 内部全局 session_manager 共享, 避免重复消息)
session_mgr = SessionManager()

# 本进程的可观测层: 会话级最近一次元信息 + 逐条消息元信息 (用于管理面板)
_last_meta: dict = {}          # session_id -> {last_intent, last_confidence, last_satisfaction, need_transfer}
_msg_meta: dict = {}            # session_id -> [ {intent, confidence, satisfaction, duration_ms, need_transfer, timestamp}, ... ] 与 history 逐条对齐


def _read_html(filename: str) -> str:
    """读取 static 目录下的 HTML,缺失时返回占位提示"""
    path = os.path.join(STATIC_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>{filename}</h1><p>静态页面 {filename} 未找到,请检查 static 目录。</p>"


# ==================== 请求模型 ====================
class ChatRequest(BaseModel):
    session_id: str = ""
    message: str


# ==================== 路由: 页面 ====================
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回用户对话界面"""
    return HTMLResponse(_read_html("index.html"))


@app.get("/admin.html", response_class=HTMLResponse)
async def admin_page():
    """返回管理面板页面"""
    return HTMLResponse(_read_html("admin.html"))


# ==================== 路由: 健康检查 ====================
@app.get("/api/health")
async def health():
    """健康检查 (不依赖 router, 独立反映服务状态)"""
    return JSONResponse({
        "status": "healthy",
        "version": getattr(settings, "APP_VERSION", "1.0.0"),
        "llm_configured": bool(getattr(settings, "LLM_API_KEY", "")),
        "modules": {
            "config": _CONFIG_OK,
            "router": _ROUTER_OK,
            "router_instance": router is not None,
            "session": _SESSION_OK,
        },
        "llm_model": getattr(settings, "LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    })


# ==================== 路由: 对话接口 ====================
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    多 Agent 对话接口
    完整流程: 意图识别 → 路由 → 子Agent(知识库/工具/闲聊/转人工) → 质检
    降级策略: core.router 不可用或运行异常时, 走关键词路由 + 质检, metadata.degraded=True
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    session_id = req.session_id or str(uuid.uuid4())[:8]

    # 1. 保存用户消息 (独立实例, 与 router 内部全局互不干扰)
    session_mgr.add_message(session_id, "user", req.message)
    _msg_meta.setdefault(session_id, []).append({
        "role": "user", "timestamp": datetime.now().isoformat(),
    })

    # 2. 执行多 Agent 流程 (router 可能为 None 或运行异常 → 降级)
    start = time.time()
    try:
        if router is not None:
            result = await router.run(session_id, req.message)
        else:
            result = _degraded_route(req.message)
    except Exception as e:
        logger.error("多Agent流程异常,降级处理: %s", e, exc_info=True)
        result = _degraded_route(req.message)
        result.setdefault("metadata", {})["error"] = str(e)
    duration_ms = int((time.time() - start) * 1000)

    reply = result.get("reply", "")
    intent = result.get("intent", "")
    confidence = result.get("intent_confidence", 0)
    satisfaction = result.get("satisfaction_score", 0)
    need_transfer = result.get("need_transfer", False)

    # 3. 归一化 metadata: 始终保证有可读 route + duration_ms
    metadata = dict(result.get("metadata", {}))
    if "route" not in metadata:
        metadata["route"] = _build_route_label(intent, degraded=metadata.get("degraded", False))
    metadata["duration_ms"] = duration_ms

    # 4. 保存 AI 回复 + 会话元信息
    session_mgr.add_message(session_id, "assistant", reply)
    _msg_meta.setdefault(session_id, []).append({
        "role": "assistant",
        "intent": intent,
        "confidence": confidence,
        "satisfaction": satisfaction,
        "duration_ms": duration_ms,
        "need_transfer": need_transfer,
        "timestamp": datetime.now().isoformat(),
    })
    _last_meta[session_id] = {
        "last_intent": intent,
        "last_confidence": confidence,
        "last_satisfaction": satisfaction,
        "need_transfer": need_transfer,
    }

    logger.info(
        "对话完成 session=%s intent=%s conf=%.2f sat=%.2f transfer=%s %dms",
        session_id, intent, confidence, satisfaction, need_transfer, duration_ms,
    )

    return JSONResponse({
        "session_id": session_id,
        "reply": reply,
        "intent": intent,
        "confidence": confidence,
        "satisfaction": satisfaction,
        "need_transfer": need_transfer,
        "metadata": metadata,
    })


# ==================== 路由: 会话管理 ====================
@app.get("/api/sessions")
async def list_sessions():
    """获取所有会话列表 (含最近一次意图/满意度)"""
    raw = session_mgr.get_all_sessions()
    sessions = []
    for s in raw:
        meta = _last_meta.get(s.get("session_id"), {})
        sessions.append({
            "session_id": s.get("session_id"),
            "message_count": s.get("message_count", 0),
            "turn_count": s.get("turn_count", 0),
            "last_message_time": s.get("last_active") or s.get("created_at"),
            "created_at": s.get("created_at"),
            "last_intent": meta.get("last_intent", ""),
            "last_confidence": meta.get("last_confidence", 0),
            "last_satisfaction": meta.get("last_satisfaction", 0),
            "need_transfer": meta.get("need_transfer", False),
        })
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定会话的对话历史 (含逐条意图/满意度元信息)"""
    history = session_mgr.get_history(session_id)
    metas = _msg_meta.get(session_id, [])
    # 逐条对齐: 末尾对齐 (历史可能被压缩, 旧的丢失元信息)
    n, m = len(history), len(metas)
    enriched = []
    for i, msg in enumerate(history):
        mi = i - (n - m)
        meta = metas[mi] if 0 <= mi < m else {}
        item = {"role": msg.get("role"), "content": msg.get("content")}
        if "timestamp" in meta:
            item["timestamp"] = meta["timestamp"]
        elif "timestamp" in msg:
            item["timestamp"] = msg.get("timestamp")
        for k in ("intent", "confidence", "satisfaction", "duration_ms", "need_transfer"):
            if k in meta:
                item[k] = meta[k]
        enriched.append(item)
    if not enriched:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或无消息")
    return JSONResponse({
        "session_id": session_id,
        "messages": enriched,
        "meta": _last_meta.get(session_id, {}),
    })


@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """清空指定会话"""
    session_mgr.clear(session_id)
    _last_meta.pop(session_id, None)
    _msg_meta.pop(session_id, None)
    return JSONResponse({
        "session_id": session_id,
        "cleared": True,
        "message": "会话已清空",
    })


# ==================== 路由: 知识库状态 ====================
@app.get("/api/knowledge/status")
async def knowledge_status():
    """知识库状态 (降级: 仅返回目录/文档统计,不依赖向量库实例)"""
    kb_dir = getattr(settings, "KB_DOCS_DIR", os.path.join(BASE_DIR, "knowledge_base"))
    if not os.path.isabs(kb_dir):
        kb_dir = os.path.join(BASE_DIR, kb_dir)
    doc_count = 0
    if os.path.isdir(kb_dir):
        doc_count = len([f for f in os.listdir(kb_dir)
                         if f.endswith((".md", ".txt", ".json"))])
    # 检测向量库相关依赖是否可用
    retriever = "keyword_fallback"
    try:
        import importlib.util as _u
        if _u.find_spec("chromadb") and _u.find_spec("sentence_transformers"):
            retriever = "vector"
    except Exception:
        pass
    return JSONResponse({
        "initialized": doc_count > 0,
        "knowledge_dir": kb_dir,
        "doc_count": doc_count,
        "module_available": _ROUTER_OK,
        "retriever": retriever,
    })


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=getattr(settings, "HOST", "0.0.0.0"),
        port=int(getattr(settings, "PORT", 8002)),
        reload=bool(getattr(settings, "DEBUG", True)),
    )
