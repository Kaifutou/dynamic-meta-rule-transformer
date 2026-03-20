from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemoryState:
    fast_slots: torch.Tensor
    slow_slots: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        fast_slots: int,
        slow_slots: int,
        d_memory: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "MemoryState":
        fast = torch.zeros(batch_size, fast_slots, d_memory, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, slow_slots, d_memory, device=device, dtype=dtype)
        return cls(fast_slots=fast, slow_slots=slow)

    def reset(self, batch_size: int | None = None) -> "MemoryState":
        if batch_size is None:
            batch_size = self.fast_slots.size(0)
        return MemoryState.zeros(
            batch_size=batch_size,
            fast_slots=self.fast_slots.size(1),
            slow_slots=self.slow_slots.size(1),
            d_memory=self.fast_slots.size(-1) if self.fast_slots.size(1) > 0 else self.slow_slots.size(-1),
            device=self.fast_slots.device,
            dtype=self.fast_slots.dtype,
        )

    def detach(self) -> "MemoryState":
        return MemoryState(
            fast_slots=self.fast_slots.detach(),
            slow_slots=self.slow_slots.detach(),
        )

    def to(self, device: torch.device | str) -> "MemoryState":
        return MemoryState(
            fast_slots=self.fast_slots.to(device),
            slow_slots=self.slow_slots.to(device),
        )
