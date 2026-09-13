"""依次写入会话并根据检索证据回答，探针答案不回写记忆。"""
from __future__ import annotations
import json
from frequency_rag.memory_agent.backends import MemoryEntry
from frequency_rag.image_selection.dataset import _list

class MemoryAgent:
    def __init__(self, memory, model):
        self.memory, self.model = memory, model

    def ingest(self, dialog):
        for session in dialog["multi_session_dialogues"]:
            for turn in session["dialogues"]:
                text = f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
                captions = [str(x) for x in _list(turn.get("image_caption"))]
                if captions:
                    text += "\nImage description: " + "\n".join(captions)
                self.memory.store(MemoryEntry(str(turn["round"]), text, turn.get("input_image", []),
                                              str(session.get("date", "")), str(session["session_id"]),
                                              [str(x) for x in _list(turn.get("image_id"))]))

    def answer(self, question, query_images=()):
        retrieved = self.memory.recall(question, query_images)
        blocks, images = [], []
        for entry in retrieved.entries:
            start = len(images) + 1
            images.extend(entry.images)
            blocks.append({"entry_id": entry.entry_id, "date": entry.timestamp,
                           "image_ids": entry.image_ids,
                           "attached_image_numbers": list(range(start, len(images) + 1)), "text": entry.text})
        query_image_numbers = list(range(len(images) + 1, len(images) + len(query_images) + 1))
        images.extend(query_images)
        prompt = ("Answer the user's question using the retrieved episodic memories. "
                  "If evidence is missing, say so. For image lookup questions use the recorded image IDs.\n"
                  "[Memory Start]\n" + json.dumps(blocks, ensure_ascii=False) + "\n[Memory End]\n"
                  + f"Question: {question}\nQuery image numbers: {query_image_numbers}")
        return self.model.complete(prompt, images), retrieved
