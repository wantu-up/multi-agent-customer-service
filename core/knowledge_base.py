"""知识库管理模块

提供客服FAQ的检索能力。优先使用chromadb做向量检索，
若chromadb不可用则降级为关键词匹配（内存版），确保系统可独立运行。
"""

try:
    import chromadb
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False


# ===== 预置10条客服FAQ文档 =====
FAQ_DOCUMENTS = [
    {
        "id": "faq_001",
        "title": "退货政策",
        "content": (
            "我们支持7天无理由退货。自签收之日起7天内，商品未使用、未拆封且包装完好，"
            "可申请退货。特殊商品（如贴身衣物、食品等）不支持无理由退货。"
            "退货流程：在「我的订单」中找到对应订单，点击「申请退货」，填写退货原因并提交，"
            "客服会在1-2个工作日内审核。审核通过后请将商品寄回，"
            "我们收到商品验收合格后3-5个工作日内退款到原支付账户。"
        ),
    },
    {
        "id": "faq_002",
        "title": "发货时间",
        "content": (
            "普通订单在付款成功后24小时内发货（节假日顺延）。"
            "预售商品按商品页面标注的发货时间发货。"
            "大促期间（如双11、618）订单量较大，发货时间可能延长至48-72小时。"
            "发货后您会收到短信通知，也可在「我的订单」中查看物流信息。"
            "如超过预计发货时间仍未发货，请联系客服催促处理。"
        ),
    },
    {
        "id": "faq_003",
        "title": "支付方式",
        "content": (
            "我们支持以下支付方式：1）微信支付；2）支付宝；3）银行卡支付（借记卡和信用卡）；"
            "4）花呗分期（满500元可用）；5）货到付款（部分城市支持）。"
            "所有支付渠道均经过加密处理，保障您的资金安全。"
            "如支付遇到问题，请先检查网络连接和账户余额，也可联系客服协助处理。"
            "分期付款免息活动请关注首页公告。"
        ),
    },
    {
        "id": "faq_004",
        "title": "会员积分",
        "content": (
            "会员积分获取方式：1）购物消费：每消费1元积1分；2）每日签到：连续签到积分递增；"
            "3）评价商品：每条有效评价积10分；4）分享商品：每次分享积5分。"
            "积分用途：1）积分兑换优惠券；2）积分兑换实物礼品；"
            "3）积分抵扣现金（100积分=1元，单笔最多抵扣订单金额的10%）。"
            "积分有效期为获取后12个月，到期未使用自动作废，请及时使用。"
        ),
    },
    {
        "id": "faq_005",
        "title": "优惠券使用",
        "content": (
            "优惠券分为满减券、折扣券和无门槛券。使用规则："
            "1）满减券需订单金额满足门槛方可使用，如满100减20；"
            "2）折扣券按比例优惠，如8折券；3）无门槛券可直接抵扣。"
            "使用方式：在结算页面「使用优惠券」中选择可用优惠券，系统自动计算优惠金额。"
            "注意事项：每个订单仅可使用一张优惠券，不可叠加使用；"
            "部分特价商品不支持优惠券；优惠券有使用期限，请在有效期内使用，过期作废不予补发。"
        ),
    },
    {
        "id": "faq_006",
        "title": "配送范围与运费",
        "content": (
            "我们配送覆盖全国（含港澳台地区，海外暂不支持）。运费标准："
            "1）满99元包邮，不足99元收取8-15元运费（按地区远近）；"
            "2）偏远地区（新疆、西藏、青海等）需补运费差价；"
            "3）大件商品运费按实际体积重量计算。大件商品配送时间通常比普通快递长2-3天。"
            "如需加急，可在下单时选择加急配送服务，加急费为15-30元。"
            "具体运费以下单页面显示为准。"
        ),
    },
    {
        "id": "faq_007",
        "title": "商品保修与售后",
        "content": (
            "电子产品享1年免费保修服务（自购买之日起），非人为损坏可免费维修。"
            "保修期内维修免收人工费和配件费，需提供订单号和保修凭证。"
            "超出保修期可选择付费维修，费用根据实际损坏情况评估。"
            "售后服务流程：联系客服报备 -> 客服确认故障 -> 寄送商品至售后中心 -> "
            "维修完成后寄回，全程通常5-10个工作日。"
            "人为损坏、进水、摔坏等不在免费保修范围内。"
        ),
    },
    {
        "id": "faq_008",
        "title": "账号安全与密码找回",
        "content": (
            "保护账号安全建议：1）设置高强度密码（字母+数字+特殊字符，8位以上）；"
            "2）不要将密码告知他人；3）定期修改密码；4）不要在公共设备上保存登录信息。"
            "忘记密码可通过以下方式找回：1）手机验证码找回：输入注册手机号，"
            "接收验证码后重置密码；2）邮箱找回：发送重置链接至绑定邮箱；"
            "3）联系客服人工验证身份后重置。"
            "如发现账号异常登录，请立即修改密码并联系客服。"
        ),
    },
    {
        "id": "faq_009",
        "title": "发票申请",
        "content": (
            "我们支持电子发票和纸质发票。电子发票在订单完成后24小时内自动开具，"
            "可在「我的订单-查看发票」中下载，与纸质发票具有同等法律效力。"
            "如需开具增值税专用发票，请在下单时备注或联系客服，"
            "需提供公司名称、税号、地址电话、开户行及账号等信息。"
            "发票抬头修改：下单时未填写发票信息的，可在订单完成后30天内联系客服补开。"
            "发票内容默认为商品明细，可改为「办公用品」等类别。"
        ),
    },
    {
        "id": "faq_010",
        "title": "尺码选择与换货",
        "content": (
            "服装鞋帽类商品提供详细尺码表和模特试穿参考，建议按尺码表选购。"
            "如尺码不合适，支持7天内换货一次（同款不同尺码，需有库存）。"
            "换货流程：在「我的订单」申请换货，选择目标尺码，审核通过后将原商品寄回，"
            "我们收到后寄出新尺码商品。换货免往返运费（尺码问题），"
            "非尺码原因换货需自理运费。每个商品限换一次，换货后不支持二次换货，但可走退货流程。"
        ),
    },
]


class KnowledgeBase:
    """知识库管理

    优先使用chromadb向量检索，不可用时降级为关键词匹配检索。
    """

    def __init__(self):
        self.docs = FAQ_DOCUMENTS
        self._chroma_client = None
        self._chroma_collection = None
        if _HAS_CHROMA:
            self._init_chroma()

    def _init_chroma(self):
        """初始化chromadb向量库（如果可用）"""
        try:
            from config import settings

            self._chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DIR
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="customer_service_kb"
            )
            # 检查是否已有数据，没有则灌入预置文档
            if self._chroma_collection.count() == 0:
                self._chroma_collection.add(
                    ids=[d["id"] for d in self.docs],
                    documents=[d["content"] for d in self.docs],
                    metadatas=[
                        {"title": d["title"], "id": d["id"]} for d in self.docs
                    ],
                )
        except Exception:
            # chromadb初始化失败，降级使用关键词匹配
            self._chroma_client = None
            self._chroma_collection = None

    def search(self, query: str, top_k: int = 3) -> list:
        """检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            相关文档列表，每项含 id、title、content、score
        """
        if self._chroma_collection is not None:
            return self._vector_search(query, top_k)
        return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list:
        """向量检索（chromadb可用时）"""
        try:
            results = self._chroma_collection.query(
                query_texts=[query], n_results=top_k
            )
            docs = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for i, doc in enumerate(documents):
                # chromadb距离越小越相关，转换为相似度分数
                score = 1 - distances[i] if distances[i] < 1 else 0
                docs.append(
                    {
                        "id": ids[i],
                        "title": metadatas[i].get("title", ""),
                        "content": doc,
                        "score": round(score, 4),
                    }
                )
            return docs
        except Exception:
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list:
        """关键词匹配检索（内存版降级方案）

        统计query中的词在文档标题和正文中出现的频率，标题匹配权重更高。
        """
        # 将查询拆分为关键词（按空格和标点分词）
        query_words = set(query.replace("，", " ").replace("。", " ").replace("？", " ").replace("?", " ").split())
        scored_docs = []
        for doc in self.docs:
            content = doc["content"]
            title = doc["title"]
            score = 0
            for word in query_words:
                if word and len(word) > 0:
                    if word in title:
                        score += 3  # 标题匹配权重更高
                    if word in content:
                        score += 1  # 正文匹配
            scored_docs.append(
                {
                    "id": doc["id"],
                    "title": title,
                    "content": content,
                    "score": score,
                }
            )
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def get_context(self, query: str) -> str:
        """返回检索结果拼接的上下文文本

        Args:
            query: 查询文本

        Returns:
            拼接后的上下文字符串，供LLM生成回答时参考
        """
        docs = self.search(query, top_k=3)
        if not docs:
            return ""
        context_parts = []
        for doc in docs:
            context_parts.append(f"【{doc['title']}】\n{doc['content']}")
        return "\n\n".join(context_parts)


# 全局知识库实例
knowledge_base = KnowledgeBase()
