import torch

from albatross_engine import sampling


def _call_sampler(logits):
    batch = logits.shape[0]
    return sampling.sample_next_tokens_batch(
        logits=logits,
        occurrence=torch.zeros_like(logits),
        temperature=torch.ones(batch, 1),
        top_p=torch.ones(batch, 1),
        top_k=torch.zeros(batch, 1, dtype=torch.long),
        alpha_presence=torch.zeros(batch, 1),
        alpha_frequency=torch.zeros(batch, 1),
        penalty_decay=torch.ones(batch, 1),
    )


def test_python_sampler_dispatch_shape(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "python")

    tokens = _call_sampler(torch.randn(4, 32))

    assert tokens.shape == (4,)
    assert tokens.dtype == torch.long


def test_greedy_sampler_dispatch_returns_argmax(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "greedy")

    tokens = _call_sampler(torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 9.0]]))

    assert tokens.tolist() == [1, 2]


def test_unknown_sampler_mode_raises(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "bad-mode")

    try:
        _call_sampler(torch.randn(2, 16))
    except ValueError as exc:
        assert "Unsupported ALBATROSS_SAMPLER" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_all_rows_equal_detects_uniform_tensor():
    assert sampling._all_rows_equal(torch.ones(4, 1))


def test_all_rows_equal_detects_non_uniform_tensor():
    assert not sampling._all_rows_equal(torch.tensor([[1.0], [2.0]]))


def test_cuda_sampler_falls_back_when_op_missing(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "cuda")
        monkeypatch.setenv("ALBATROSS_SAMPLER_FALLBACK", "1")
        monkeypatch.setattr(sampling, "_cuda_sampler_available", lambda: False)
    else:
        original_available = getattr(sampling, "_cuda_sampler_available", None)
        sampling._cuda_sampler_available = lambda: False

    try:
        tokens = _call_sampler(torch.randn(2, 16))
        assert tokens.shape == (2,)
    finally:
        if monkeypatch is None and original_available is not None:
            sampling._cuda_sampler_available = original_available


def test_cuda_sampler_falls_back_for_non_uniform_parameters(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "cuda")
        monkeypatch.setenv("ALBATROSS_SAMPLER_FALLBACK", "1")
        monkeypatch.setattr(sampling, "_cuda_sampler_available", lambda: True)
        monkeypatch.setattr(
            sampling,
            "_sample_cuda_uniform",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call cuda")),
        )
    else:
        original_available = sampling._cuda_sampler_available
        original_cuda = sampling._sample_cuda_uniform
        sampling._cuda_sampler_available = lambda: True
        sampling._sample_cuda_uniform = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not call cuda")
        )

    try:
        logits = torch.randn(2, 16)
        tokens = sampling.sample_next_tokens_batch(
            logits=logits,
            occurrence=torch.zeros_like(logits),
            temperature=torch.tensor([[1.0], [0.8]]),
            top_p=torch.ones(2, 1),
            top_k=torch.zeros(2, 1, dtype=torch.long),
            alpha_presence=torch.zeros(2, 1),
            alpha_frequency=torch.zeros(2, 1),
            penalty_decay=torch.ones(2, 1),
        )

        assert tokens.shape == (2,)
    finally:
        if monkeypatch is None:
            sampling._cuda_sampler_available = original_available
            sampling._sample_cuda_uniform = original_cuda


def test_cuda_sampler_allows_non_uniform_penalty_tensors(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("ALBATROSS_SAMPLER", "cuda")
        monkeypatch.setenv("ALBATROSS_SAMPLER_FALLBACK", "0")
        monkeypatch.setattr(sampling, "_cuda_sampler_available", lambda: True)
        monkeypatch.setattr(sampling, "_sample_cuda_uniform", lambda *args, **kwargs: torch.tensor([1, 2]))
    else:
        original_available = sampling._cuda_sampler_available
        original_cuda = sampling._sample_cuda_uniform
        sampling._cuda_sampler_available = lambda: True
        sampling._sample_cuda_uniform = lambda *args, **kwargs: torch.tensor([1, 2])

    try:
        logits = torch.randn(2, 16)
        occurrence = torch.zeros_like(logits)
        occurrence[1, 3] = 1.0
        tokens = sampling.sample_next_tokens_batch(
            logits=logits,
            occurrence=occurrence,
            temperature=torch.ones(2, 1),
            top_p=torch.ones(2, 1),
            top_k=torch.zeros(2, 1, dtype=torch.long),
            alpha_presence=occurrence.clone(),
            alpha_frequency=torch.ones(2, 1),
            penalty_decay=torch.tensor([[0.996], [0.5]]),
        )

        assert tokens.tolist() == [1, 2]
    finally:
        if monkeypatch is None:
            sampling._cuda_sampler_available = original_available
            sampling._sample_cuda_uniform = original_cuda
