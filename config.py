"""全局配置模块

使用 python-dotenv 加载 .env 文件，集中管理多Agent智能客服系统的所有配置项。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Settings:
    """全局配置类

    所有配置项集中管理，支持通过环境变量覆盖默认值。
    """

    # ===== 应用基本信息 =====
    APP_NAME = "多Agent智能客服系统"
    APP_VERSION = "1.0.0"

    # ===== 服务配置 =====
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8002"))

    # ===== LLM 配置 =====
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 2000

    # ===== 数据库配置 =====
    DB_PATH = os.getenv("DB_PATH", "./data/customer_service.db")

    # ===== 对话配置 =====
    MAX_HISTORY_TURNS = 10  # 对话历史最大轮数

    # ===== 质检/转人工配置 =====
    RISK_KEYWORDS = ["投诉", "差评", "退款", "举报", "律师"]  # 触发转人工的关键词
    SATISFACTION_THRESHOLD = 0.6  # 满意度低于此值触发转人工

    # ===== 知识库配置 =====
    KB_DOCS_DIR = os.getenv("KB_DOCS_DIR", "./knowledge_base")
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")


# 全局配置实例
settings = Settings()
