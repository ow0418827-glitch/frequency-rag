"""共享的向量数值运算。"""
import numpy as np

def normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)
