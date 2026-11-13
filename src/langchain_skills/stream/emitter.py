"""
StreamEventEmitter - 统一事件格式

所有事件都包含 type 和相关数据。
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StreamEvent:
    """统一的流式事件"""
    type: str
    data: Dict[str, Any]


class StreamEventEmitter:
    """流式事件发射器"""

    @staticmethod
    def thinking(content: str, thinking_id: int = 0) -> StreamEvent:
        """思考内容事件"""
        return StreamEvent("thinking", {"type": "thinking", "content": content, "id": thinking_id})

    @staticmethod
    def text(content: str) -> StreamEvent:
        """文本内容事件"""
        return StreamEvent("text", {"type": "text", "content": content})

    @staticmethod
    def tool_call(name: str, args: Dict[str, Any], tool_id: str = "") -> StreamEvent:
        """工具调用事件"""
        return StreamEvent("tool_call", {"type": "tool_call", "name": name, "args": args, "id": tool_id})

    @staticmethod
    def tool_result(name: str, content: str, success: bool = True) -> StreamEvent:
        """工具结果事件"""
        return StreamEvent("tool_result", {
            "type": "tool_result",
            "name": name,
            "content": content,
            "success": success,
        })

    @staticmethod
    def done(response: str = "", structured: dict | None = None) -> StreamEvent:
        """完成事件（任务型时附带结构化交付报告）"""
        data: dict = {"type": "done", "response": response}
        if structured is not None:
            data["structured"] = structured
        return StreamEvent("done", data)

    @staticmethod
    def interrupt(interrupts: list[dict]) -> StreamEvent:
        """人工审批事件（HITL）：流在这里暂停，等待客户端 resume"""
        return StreamEvent("interrupt", {"type": "interrupt", "interrupts": interrupts})

    @staticmethod
    def error(message: str) -> StreamEvent:
        """错误事件"""
        return StreamEvent("error", {"type": "error", "message": message})
