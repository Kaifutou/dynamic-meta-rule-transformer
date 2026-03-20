from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DemoConfig
from .memory import MemoryState


@dataclass
class VariantFlags:
    name: str
    dynamic_enabled: bool
    use_fast_memory: bool
    use_slow_memory: bool
    allow_memory_write: bool


class DynamicLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        self.A = nn.Parameter(torch.empty(out_features, rank))
        nn.init.xavier_uniform_(self.A)

    def forward(self, x: torch.Tensor, b_dynamic: torch.Tensor | None) -> torch.Tensor:
        out = self.base(x)
        if b_dynamic is None:
            return out
        delta = torch.einsum("bti,bri,or->bto", x, b_dynamic, self.A)
        return out + delta


class DynamicTransformerLayer(nn.Module):
    def __init__(self, config: DemoConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.q_proj = DynamicLinear(self.d_model, self.d_model, config.lora_rank, bias=False)
        self.k_proj = DynamicLinear(self.d_model, self.d_model, config.lora_rank, bias=False)
        self.v_proj = DynamicLinear(self.d_model, self.d_model, config.lora_rank, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.ff1 = DynamicLinear(self.d_model, config.d_ff, config.lora_rank, bias=True)
        self.ff2 = DynamicLinear(config.d_ff, self.d_model, config.lora_rank, bias=True)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, dyn: dict[str, torch.Tensor | None]) -> torch.Tensor:
        residual = x
        h = self.norm1(x)

        q = self.q_proj(h, dyn.get("q"))
        k = self.k_proj(h, dyn.get("k"))
        v = self.v_proj(h, dyn.get("v"))

        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(score, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        x = residual + self.dropout(self.o_proj(ctx))

        residual = x
        h = self.norm2(x)
        ff = F.gelu(self.ff1(h, dyn.get("ff1")))
        ff = self.ff2(ff, dyn.get("ff2"))
        x = residual + self.dropout(ff)
        return x


class MetaRuleController(nn.Module):
    def __init__(
        self,
        config: DemoConfig,
        fast_slots: int,
        slow_slots: int,
    ):
        super().__init__()
        self.config = config
        self.fast_slots = fast_slots
        self.slow_slots = slow_slots

        self.q_read = nn.Linear(config.d_model, config.d_memory, bias=False)
        self.k_read = nn.Linear(config.d_memory, config.d_memory, bias=False)
        self.v_read = nn.Linear(config.d_memory, config.d_memory, bias=False)

        ctrl_in = config.d_model + config.d_memory
        self.ctrl_mlp = nn.Sequential(
            nn.Linear(ctrl_in, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )

        self.per_layer_dynamic = config.lora_rank * (4 * config.d_model + config.d_ff)
        self.dynamic_heads = nn.ModuleList(
            nn.Linear(config.d_model, self.per_layer_dynamic) for _ in range(config.n_layers)
        )

        feature_dim = config.d_model + config.d_memory + 3
        self.z_proj = nn.Linear(feature_dim, config.d_memory)
        self.alpha_proj = nn.Linear(feature_dim, 1)
        self.rho_proj = nn.Linear(feature_dim, 1)

        self.fast_slot_proj = nn.Linear(feature_dim, fast_slots) if fast_slots > 0 else None
        self.slow_slot_proj = nn.Linear(feature_dim, slow_slots) if slow_slots > 0 else None
        self.c_proj = nn.Linear(config.d_memory, config.d_memory)

        self.register_buffer("lambda_fast", torch.tensor(config.lambda_fast, dtype=torch.float32))

    def read_memory(self, state: MemoryState, context: torch.Tensor) -> torch.Tensor:
        memories = []
        if state.fast_slots.size(1) > 0:
            memories.append(state.fast_slots)
        if state.slow_slots.size(1) > 0:
            memories.append(state.slow_slots)
        if not memories:
            return torch.zeros(
                context.size(0),
                self.config.d_memory,
                device=context.device,
                dtype=context.dtype,
            )

        mem = torch.cat(memories, dim=1)
        q = self.q_read(context)
        k = self.k_read(mem)
        v = self.v_read(mem)

        score = torch.einsum("bd,bsd->bs", q, k) / math.sqrt(self.config.d_memory)
        weight = torch.softmax(score, dim=-1)
        readout = torch.einsum("bs,bsd->bd", weight, v)
        return readout

    def generate_dynamic(
        self,
        context: torch.Tensor,
        readout: torch.Tensor,
        enabled: bool,
        force_zero: bool,
    ) -> tuple[list[dict[str, torch.Tensor | None]], torch.Tensor]:
        if not enabled or force_zero:
            return [self._empty_dyn() for _ in range(self.config.n_layers)], torch.zeros(
                (), device=context.device, dtype=context.dtype
            )

        ctrl = self.ctrl_mlp(torch.cat([context, readout], dim=-1))
        layers_dyn: list[dict[str, torch.Tensor | None]] = []
        norms = []
        r = self.config.lora_rank

        for head in self.dynamic_heads:
            raw = torch.tanh(head(ctrl)) * self.config.dynamic_scale
            offset = 0

            q_size = r * self.config.d_model
            k_size = r * self.config.d_model
            v_size = r * self.config.d_model
            ff1_size = r * self.config.d_model
            ff2_size = r * self.config.d_ff

            b_q = raw[:, offset : offset + q_size].view(-1, r, self.config.d_model)
            offset += q_size
            b_k = raw[:, offset : offset + k_size].view(-1, r, self.config.d_model)
            offset += k_size
            b_v = raw[:, offset : offset + v_size].view(-1, r, self.config.d_model)
            offset += v_size
            b_ff1 = raw[:, offset : offset + ff1_size].view(-1, r, self.config.d_model)
            offset += ff1_size
            b_ff2 = raw[:, offset : offset + ff2_size].view(-1, r, self.config.d_ff)

            layers_dyn.append(
                {
                    "q": b_q,
                    "k": b_k,
                    "v": b_v,
                    "ff1": b_ff1,
                    "ff2": b_ff2,
                }
            )
            norms.extend([b_q, b_k, b_v, b_ff1, b_ff2])

        delta_norm = torch.stack([n.pow(2).mean() for n in norms]).mean()
        return layers_dyn, delta_norm

    def update_state(
        self,
        state: MemoryState,
        context: torch.Tensor,
        readout: torch.Tensor,
        logits: torch.Tensor,
        feedback: dict[str, torch.Tensor] | None,
        allow_write: bool,
        use_fast_memory: bool,
        use_slow_memory: bool,
    ) -> tuple[MemoryState, dict[str, torch.Tensor]]:
        device = logits.device
        bsz = logits.size(0)

        if not allow_write or (state.fast_slots.size(1) == 0 and state.slow_slots.size(1) == 0):
            zeros = torch.zeros(bsz, device=device)
            return state, {"alpha": zeros, "rho": zeros}

        prob = torch.softmax(logits, dim=-1)
        max_prob = prob.max(dim=-1).values.unsqueeze(-1)
        entropy = (-(prob * prob.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True))

        if feedback is not None and "targets" in feedback:
            targets = feedback["targets"].long()
            target_prob = prob.gather(1, targets.unsqueeze(1))
            error = 1.0 - target_prob
        else:
            error = torch.zeros(bsz, 1, device=device, dtype=prob.dtype)

        feat = torch.cat([context, readout, max_prob, entropy, error], dim=-1)
        z = torch.tanh(self.z_proj(feat))
        alpha = torch.sigmoid(self.alpha_proj(feat))

        fast_next = state.fast_slots
        if use_fast_memory and state.fast_slots.size(1) > 0 and self.fast_slot_proj is not None:
            fast_w = torch.softmax(self.fast_slot_proj(feat), dim=-1)
            add_fast = fast_w.unsqueeze(-1) * z.unsqueeze(1)
            fast_next = self.lambda_fast * state.fast_slots + alpha.unsqueeze(-1) * add_fast

        rho = torch.zeros(bsz, 1, device=device, dtype=prob.dtype)
        slow_next = state.slow_slots
        if use_slow_memory and state.slow_slots.size(1) > 0 and self.slow_slot_proj is not None:
            rho = torch.sigmoid(self.rho_proj(feat))
            if fast_next.size(1) > 0:
                memory_summary = fast_next.mean(dim=1)
            else:
                memory_summary = z
            candidate = torch.tanh(self.c_proj(memory_summary))
            slow_w = torch.softmax(self.slow_slot_proj(feat), dim=-1)
            consolidated = slow_w.unsqueeze(-1) * candidate.unsqueeze(1)
            slow_next = (1.0 - rho.unsqueeze(-1)) * state.slow_slots + rho.unsqueeze(-1) * consolidated

        return MemoryState(fast_slots=fast_next, slow_slots=slow_next), {
            "alpha": alpha.squeeze(-1),
            "rho": rho.squeeze(-1),
        }

    @staticmethod
    def _empty_dyn() -> dict[str, torch.Tensor | None]:
        return {"q": None, "k": None, "v": None, "ff1": None, "ff2": None}


class MetaRuleTransformer(nn.Module):
    def __init__(
        self,
        config: DemoConfig,
        variant: VariantFlags,
    ):
        super().__init__()
        self.config = config
        self.variant = variant

        self.fast_slots = config.fast_slots if variant.use_fast_memory else 0
        self.slow_slots = config.slow_slots if variant.use_slow_memory else 0

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(DynamicTransformerLayer(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

        self.controller = MetaRuleController(config, fast_slots=self.fast_slots, slow_slots=self.slow_slots)
        self.memory_adapter = nn.Linear(config.d_memory, config.d_model, bias=False)

        self.force_zero_dynamic = False

    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> MemoryState:
        device = device or self.config.device
        return MemoryState.zeros(
            batch_size=batch_size,
            fast_slots=self.fast_slots,
            slow_slots=self.slow_slots,
            d_memory=self.config.d_memory,
            device=device,
        )

    def set_force_zero_dynamic(self, value: bool) -> None:
        self.force_zero_dynamic = value

    def forward(
        self,
        tokens: torch.Tensor,
        state: MemoryState | None,
        feedback: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, MemoryState, dict[str, torch.Tensor]]:
        bsz, seq_len = tokens.shape
        if seq_len != self.config.seq_len:
            raise ValueError(f"Expected seq_len={self.config.seq_len}, got {seq_len}")

        if state is None or state.fast_slots.size(0) != bsz:
            state = self.init_state(batch_size=bsz, device=tokens.device)

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        base_context = x.mean(dim=1)
        readout = self.controller.read_memory(state, base_context)
        x = x + self.memory_adapter(readout).unsqueeze(1)
        x = self.drop(x)

        dyn_context = x.mean(dim=1)
        dynamic_layers, delta_norm = self.controller.generate_dynamic(
            context=dyn_context,
            readout=readout,
            enabled=self.variant.dynamic_enabled,
            force_zero=self.force_zero_dynamic,
        )

        for layer, dyn in zip(self.layers, dynamic_layers):
            x = layer(x, dyn)

        x = self.norm(x)
        pooled = x[:, -1, :]
        logits = self.head(pooled)

        next_state, gate_aux = self.controller.update_state(
            state=state,
            context=pooled,
            readout=readout,
            logits=logits,
            feedback=feedback,
            allow_write=self.variant.allow_memory_write,
            use_fast_memory=self.variant.use_fast_memory,
            use_slow_memory=self.variant.use_slow_memory,
        )

        aux = {
            "delta_norm": delta_norm,
            "alpha": gate_aux["alpha"],
            "rho": gate_aux["rho"],
        }
        return logits, next_state, aux
