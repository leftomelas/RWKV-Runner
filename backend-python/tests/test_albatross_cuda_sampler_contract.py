import torch


def _has_cuda_sampler():
    return (
        torch.cuda.is_available()
        and hasattr(torch.ops, "rwkv7_state_fwd_fp16")
        and hasattr(torch.ops.rwkv7_state_fwd_fp16, "setup_rand")
        and hasattr(torch.ops.rwkv7_state_fwd_fp16, "batch_sampling_temperature_topk_topp")
        and hasattr(torch.ops.rwkv7_state_fwd_fp16, "batch_sampling_repetition_temperature_topk_topp")
    )


def test_cuda_sampler_contract_if_loaded():
    if torch.cuda.is_available():
        import albatross.rwkv7  # noqa: F401

    if not torch.cuda.is_available() or not _has_cuda_sampler():
        return

    logits = torch.randn(8, 65536, device="cuda", dtype=torch.float32)
    penalties = torch.zeros_like(logits)
    states = torch.ops.rwkv7_state_fwd_fp16.setup_rand(123, 8)

    tokens = torch.ops.rwkv7_state_fwd_fp16.batch_sampling_repetition_temperature_topk_topp(
        logits,
        penalties,
        states,
        0.0,
        1.0,
        0.996,
        1.0,
        0,
        0.3,
    )

    assert tokens.shape == (8,)
    assert tokens.device.type == "cuda"
    assert tokens.dtype in (torch.int32, torch.int64)
