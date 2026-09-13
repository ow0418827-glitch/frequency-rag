"""神经图记忆：存储时连接旧节点，查询时按相似度有界遍历。"""
from .vector import VectorMemory


class NGMemory(VectorMemory):
    kind = "ngmemory"
    graph_enabled = True
    edge_threshold = 0.6
    traversal_threshold = 0.4
