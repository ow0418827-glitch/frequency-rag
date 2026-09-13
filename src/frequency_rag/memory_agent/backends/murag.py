"""平面多模态记忆：直接使用共享余弦检索，不建图、不调用路由模型。"""
from .vector import VectorMemory


class MuRAGMemory(VectorMemory):
    kind = "murag"
