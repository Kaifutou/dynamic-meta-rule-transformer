from __future__ import annotations

import torch

from metarule_demo.config import DemoConfig
from metarule_demo.interactive import _build_tokens, _decode_tokens, _parse_supports


def test_parse_supports_ok() -> None:
    pairs = _parse_supports("1->2, 3->4, 5->6, 7->8", expected=4, vocab_symbols=32)
    assert pairs == [(1, 2), (3, 4), (5, 6), (7, 8)]


def test_build_tokens_shape() -> None:
    cfg = DemoConfig(support_size=4, vocab_symbols=32)
    tokens = _build_tokens(cfg, pairs=[(1, 2), (3, 4), (5, 6), (7, 8)], query=9, device=torch.device("cpu"))
    assert tokens.shape == (1, cfg.seq_len)


def test_build_decode_roundtrip() -> None:
    cfg = DemoConfig(support_size=4, vocab_symbols=32)
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
    query = 9
    tokens = _build_tokens(cfg, pairs=pairs, query=query, device=torch.device("cpu"))
    got_pairs, got_query = _decode_tokens(cfg, tokens[0])
    assert got_pairs == pairs
    assert got_query == query
