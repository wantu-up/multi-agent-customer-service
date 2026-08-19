"""客服工具集模块

模拟3个客服工具（查询订单、查询物流、申请退款），使用mock数据，
不连接真实系统，便于演示和测试。
"""

import uuid
from datetime import datetime, timedelta


# ===== Mock 订单数据库 =====
MOCK_ORDERS = {
    "ORD20240001": {
        "order_id": "ORD20240001",
        "status": "已发货",
        "logistics_company": "顺丰速运",
        "tracking_no": "SF1234567890",
        "logistics_status": "运输中",
        "estimated_arrival": "2024-12-20",
        "amount": 299.00,
        "items": "无线蓝牙耳机 x1",
        "created_at": "2024-12-15 10:30:00",
    },
    "ORD20240002": {
        "order_id": "ORD20240002",
        "status": "待发货",
        "logistics_company": "中通快递",
        "tracking_no": "ZTO9876543210",
        "logistics_status": "未揽收",
        "estimated_arrival": "2024-12-22",
        "amount": 159.00,
        "items": "手机壳 x2",
        "created_at": "2024-12-16 14:20:00",
    },
    "ORD20240003": {
        "order_id": "ORD20240003",
        "status": "已完成",
        "logistics_company": "京东物流",
        "tracking_no": "JD5678901234",
        "logistics_status": "已签收",
        "estimated_arrival": "2024-12-18",
        "amount": 899.00,
        "items": "智能手表 x1",
        "created_at": "2024-12-10 09:15:00",
    },
    "ORD20240004": {
        "order_id": "ORD20240004",
        "status": "已取消",
        "logistics_company": "圆通速递",
        "tracking_no": "YT2468101357",
        "logistics_status": "已退回",
        "estimated_arrival": "—",
        "amount": 49.00,
        "items": "数据线 x3",
        "created_at": "2024-12-08 16:45:00",
    },
    "ORD20240005": {
        "order_id": "ORD20240005",
        "status": "已发货",
        "logistics_company": "韵达快递",
        "tracking_no": "YD3692581470",
        "logistics_status": "到达派送站",
        "estimated_arrival": "2024-12-19",
        "amount": 1299.00,
        "items": "机械键盘 x1",
        "created_at": "2024-12-14 11:00:00",
    },
}


class CustomerTools:
    """客服工具集

    提供3个常用客服工具，均使用mock数据模拟，便于独立运行和演示。
    工具描述使用中文，方便LLM理解和调用。
    """

    # 工具描述（供LLM function calling使用）
    TOOL_DESCRIPTIONS = [
        {
            "name": "check_order",
            "description": "查询订单状态。根据订单号查询订单的当前状态、商品信息和金额。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD20240001",
                    }
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "check_logistics",
            "description": "查询物流信息。根据订单号查询快递公司、运单号和物流状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD20240001",
                    }
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "request_refund",
            "description": "申请退款。根据订单号和退款原因发起退款申请。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD20240001",
                    },
                    "reason": {
                        "type": "string",
                        "description": "退款原因",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    ]

    async def check_order(self, order_id: str) -> dict:
        """查询订单状态

        Args:
            order_id: 订单编号

        Returns:
            包含订单详细信息的字典，含 status 和 message 字段
        """
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return {
                "status": "error",
                "message": f"未找到订单号为 {order_id} 的订单，请确认订单号是否正确。",
            }
        return {
            "status": "success",
            "message": f"订单 {order_id} 查询成功",
            "data": {
                "order_id": order["order_id"],
                "status": order["status"],
                "items": order["items"],
                "amount": f"¥{order['amount']:.2f}",
                "created_at": order["created_at"],
            },
        }

    async def check_logistics(self, order_id: str) -> dict:
        """查询物流信息

        Args:
            order_id: 订单编号

        Returns:
            包含物流详细信息的字典，含 status 和 message 字段
        """
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return {
                "status": "error",
                "message": f"未找到订单号为 {order_id} 的订单，无法查询物流信息。",
            }
        return {
            "status": "success",
            "message": f"订单 {order_id} 物流查询成功",
            "data": {
                "order_id": order["order_id"],
                "logistics_company": order["logistics_company"],
                "tracking_no": order["tracking_no"],
                "logistics_status": order["logistics_status"],
                "estimated_arrival": order["estimated_arrival"],
            },
        }

    async def request_refund(self, order_id: str, reason: str) -> dict:
        """申请退款

        Args:
            order_id: 订单编号
            reason: 退款原因

        Returns:
            包含退款申请结果的字典，含 status 和 message 字段
        """
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return {
                "status": "error",
                "message": f"未找到订单号为 {order_id} 的订单，无法申请退款。",
            }
        # 已完成或已取消的订单不支持退款（演示逻辑）
        if order["status"] in ("已完成", "已取消"):
            return {
                "status": "error",
                "message": f"订单 {order_id} 当前状态为「{order['status']}」，暂不支持退款。",
            }
        refund_no = f"RF{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
        estimated_refund = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        return {
            "status": "success",
            "message": f"订单 {order_id} 退款申请已提交",
            "data": {
                "refund_no": refund_no,
                "order_id": order_id,
                "reason": reason,
                "refund_amount": f"¥{order['amount']:.2f}",
                "process_status": "审核中",
                "estimated_refund_date": estimated_refund,
            },
        }


# 全局工具实例
customer_tools = CustomerTools()
