from __future__ import annotations

import copy

import torch

from metarule_demo.config import DemoConfig
from metarule_demo.memory import MemoryState
from metarule_demo.model import MetaRuleTransformer, VariantFlags


def tiny_config() -> DemoConfig:
    return DemoConfig(
        vocab_symbols=12,
        support_size=3,
        num_rules=8,
        phase_rule_count=2,
        phase_sequence=(0, 1, 0),
        batch_size=4,
        eval_batch_size=4,
        online_batch_size=4,
        train_steps=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        lora_rank=2,
        d_memory=16,
        fast_slots=4,
        slow_slots=6,
        dropout=0.0,
    )


def test_memory_reset_and_detach() -> None:
    state = MemoryState.zeros(
        batch_size=3,
        fast_slots=2,
        slow_slots=3,
        d_memory=8,
        device="cpu",
    )
    state.fast_slots += 1.0
    state.slow_slots += 2.0

    reset_state = state.reset()
    assert reset_state.fast_slots.shape == (3, 2, 8)
    assert reset_state.slow_slots.shape == (3, 3, 8)
    assert torch.allclose(reset_state.fast_slots, torch.zeros_like(reset_state.fast_slots))
    assert torch.allclose(reset_state.slow_slots, torch.zeros_like(reset_state.slow_slots))

    detached = state.detach()
    assert detached.fast_slots.requires_grad is False
    assert detached.slow_slots.requires_grad is False


def test_forward_shapes_gate_ranges_and_nan() -> None:
    cfg = tiny_config()
    variant = VariantFlags(
        name="full",
        dynamic_enabled=True,
        use_fast_memory=True,
        use_slow_memory=True,
        allow_memory_write=True,
    )
    model = MetaRuleTransformer(cfg, variant)

    batch_size = 4
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (batch_size,))

    state = model.init_state(batch_size)
    logits, next_state, aux = model(tokens, state, feedback={"targets": targets})

    assert logits.shape == (batch_size, cfg.vocab_size)
    assert not torch.isnan(next_state.fast_slots).any()
    assert not torch.isnan(next_state.slow_slots).any()

    assert torch.all(aux["alpha"] >= 0.0)
    assert torch.all(aux["alpha"] <= 1.0)
    assert torch.all(aux["rho"] >= 0.0)
    assert torch.all(aux["rho"] <= 1.0)
    assert float(aux["delta_norm"].item()) >= 0.0


def test_disable_dynamic_and_write_matches_static_behavior() -> None:
    torch.manual_seed(7)
    cfg = tiny_config()

    base_variant = VariantFlags(
        name="dynamic_but_zero",
        dynamic_enabled=True,
        use_fast_memory=False,
        use_slow_memory=False,
        allow_memory_write=False,
    )
    model_dyn = MetaRuleTransformer(cfg, base_variant)
    model_dyn.eval()
    model_dyn.set_force_zero_dynamic(True)

    model_static = copy.deepcopy(model_dyn)
    model_static.variant.dynamic_enabled = False

    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.seq_len))
    state_dyn = model_dyn.init_state(3)
    state_static = model_static.init_state(3)

    with torch.no_grad():
        logits_dyn, next_dyn, _ = model_dyn(tokens, state_dyn, feedback=None)
        logits_static, next_static, _ = model_static(tokens, state_static, feedback=None)

    assert torch.allclose(logits_dyn, logits_static, atol=1e-6)
    assert torch.allclose(next_dyn.fast_slots, next_static.fast_slots, atol=1e-6)
    assert torch.allclose(next_dyn.slow_slots, next_static.slow_slots, atol=1e-6)
