from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any
import weakref

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class PhaseClock:
    device: torch.device
    started_at: float | None = None

    def start(self) -> None:
        synchronize(self.device)
        self.started_at = time.perf_counter()

    def stop(self) -> float:
        if self.started_at is None:
            raise RuntimeError("计时器尚未启动。")
        synchronize(self.device)
        elapsed = time.perf_counter() - self.started_at
        self.started_at = None
        return elapsed


class DeviceMemoryMonitor:
    """同时记录框架峰值和设备总占用采样；处理器运行时返回空显存字段。"""

    def __init__(self, device: torch.device, sample_interval_seconds: float = 0.05) -> None:
        self.device = device
        self.sample_interval_seconds = max(0.01, float(sample_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_used: int | None = None
        self._maximum_used: int | None = None
        self._sample_count = 0
        self._sampling_error: str | None = None
        self._started = False
        self._stopped = False

    def _read_device_used(self) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        return int(total - free)

    @staticmethod
    def _sample_loop(reference: "weakref.ReferenceType[DeviceMemoryMonitor]") -> None:
        while True:
            owner = reference()
            if owner is None:
                return
            stop_event = owner._stop
            interval = owner.sample_interval_seconds
            del owner
            if stop_event.wait(interval):
                return
            owner = reference()
            if owner is None:
                return
            try:
                used = owner._read_device_used()
                owner._sample_count += 1
                owner._maximum_used = (
                    used if owner._maximum_used is None else max(owner._maximum_used, used)
                )
            except BaseException as exc:  # 采样失败不应掩盖主实验结果。
                owner._sampling_error = f"{type(exc).__name__}: {exc}"
                return
            finally:
                del owner

    def start(self) -> None:
        if self._started:
            raise RuntimeError("显存监视器不能重复启动。")
        self._started = True
        if self.device.type != "cuda":
            return
        synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._baseline_used = self._read_device_used()
        self._maximum_used = self._baseline_used
        self._sample_count = 1
        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(weakref.ref(self),),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("显存监视器尚未启动。")
        if self._stopped:
            raise RuntimeError("显存监视器已经停止。")
        if self.device.type != "cuda":
            self._stop.set()
            self._stopped = True
            return {
                "device": str(self.device),
                "framework_peak_allocated_bytes": None,
                "framework_peak_reserved_bytes": None,
                "device_used_baseline_bytes": None,
                "device_used_peak_sampled_bytes": None,
                "device_sample_interval_seconds": None,
                "device_sample_count": 0,
                "device_sampling_error": None,
            }
        synchronize(self.device)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        try:
            final_used = self._read_device_used()
            self._sample_count += 1
            self._maximum_used = max(self._maximum_used or 0, final_used)
        except BaseException as exc:
            self._sampling_error = self._sampling_error or f"{type(exc).__name__}: {exc}"
        self._stopped = True
        return {
            "device": str(self.device),
            "framework_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "framework_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            "device_used_baseline_bytes": self._baseline_used,
            "device_used_peak_sampled_bytes": self._maximum_used,
            "device_sample_interval_seconds": self.sample_interval_seconds,
            "device_sample_count": self._sample_count,
            "device_sampling_error": self._sampling_error,
        }

    def cancel(self) -> None:
        """异常路径只终止采样线程，不再读取设备或生成测量结果。"""
        if not self._started or self._stopped:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self._stopped = True

    def __del__(self) -> None:
        # 异常退出时只终止后台采样；析构阶段不再调用图形处理器接口。
        self._stop.set()
