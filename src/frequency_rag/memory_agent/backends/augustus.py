"""概念标签图记忆：概念提取、共享概念建边和双通道排序。"""
import numpy as np
from .vector import VectorMemory


class AugustusMemory(VectorMemory):
    kind = "augustus"
    graph_enabled = True
    edge_threshold = 0.7
    traversal_threshold = 0.5

    def extract_tags(self, text):
        result = self.llm.complete('Extract semantic keyword concepts from the following text. '
                                   'Return JSON {"concepts": ["..."]}.\n' + text, structured=True)
        concepts = result.get("concepts")
        if not isinstance(concepts, list) or any(not isinstance(x, str) for x in concepts):
            raise ValueError("概念提取响应格式错误。")
        return {x.strip().lower() for x in concepts if x.strip()}

    def link_allowed(self, tags, previous_tags):
        return bool(tags & previous_tags)

    def rank_scores(self, question, cosines):
        tags = self.extract_tags(question)
        coverage = np.array([len(tags & t) / max(len(tags), 1) for t in self.tags])
        return .5 * cosines + .5 * coverage
