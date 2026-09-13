"""模态路由记忆：语言模型决定无检索、文本检索或图像检索。"""
from .vector import VectorMemory


class UniversalRAGMemory(VectorMemory):
    kind = "universalrag"

    def route_query(self, question, images):
        result = self.llm.complete('Choose memory retrieval channel for this query: no, document, image. '
                                   'Return JSON {"route": "..."}.\n' + question, images, structured=True)
        route = result.get("route")
        route = {"paragraph": "document", "clip": "image"}.get(route, route)
        if route not in {"no", "document", "image"}:
            raise ValueError("无效的检索路由。")
        return route
