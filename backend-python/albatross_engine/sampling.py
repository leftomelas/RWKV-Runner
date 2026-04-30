import os
from typing import List, Tuple, Dict, Any, Union

import torch
from torch.nn import functional as F

from collections import defaultdict

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop


@MyStatic
def sample_logits_real_batch(
    logits: torch.Tensor,
    temperature: torch.Tensor,  # [bsz, 1]
    top_p: torch.Tensor,  # [bsz, 1]
    top_k: torch.Tensor,  # [bsz, 1]
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    bsz, vocab_size = logits.shape

    # ====== 1. 温度缩放 (完全并行) ======
    # 直接广播 [bsz, 1] 的 temperature 到 [bsz, vocab_size]
    logits = logits / temperature

    # ====== 2. 计算概率分布 ======
    probs = F.softmax(logits.float(), dim=-1)

    # ====== 3. Top-k 过滤 (完全向量化) ======
    # 创建有效样本掩码 (top_k > 0)
    active_top_k = top_k > 0  # [bsz, 1]

    if active_top_k.any():
        # 调整 top_k 值: [min_tokens_to_keep, vocab_size]
        clamped_top_k = torch.clamp(top_k, min=min_tokens_to_keep, max=vocab_size).long()  # [bsz, 1]

        # 为无效样本设置虚拟值 (vocab_size 表示不过滤)
        effective_top_k = torch.where(
            active_top_k, clamped_top_k, torch.full_like(clamped_top_k, vocab_size)
        )  # [bsz, 1]

        # 生成排序索引和概率 [bsz, vocab_size]
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # 生成排名矩阵 [0, 1, ..., vocab_size-1] -> [bsz, vocab_size]
        ranks = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size)

        # 创建移除掩码: 排名 >= effective_top_k 的位置
        remove_mask = ranks >= effective_top_k  # 广播 [bsz, 1] -> [bsz, vocab_size]

        # 应用掩码: 将低概率位置置零
        sorted_probs[remove_mask] = 0.0

        # 重建原始顺序的概率分布
        filtered_probs = torch.zeros_like(probs).scatter_(dim=1, index=sorted_indices, src=sorted_probs)

        # 更新概率 (仅活跃样本被修改)
        probs = torch.where(active_top_k.expand_as(probs), filtered_probs, probs)

    # ====== 4. Top-p 过滤 (完全向量化) ======
    # 为 top_p >= 1.0 的样本创建虚拟阈值 (2.0 确保不触发过滤)
    effective_top_p = torch.where(top_p < 1.0, top_p, torch.full_like(top_p, 2.0))  # [bsz, 1]

    # 生成排序索引和概率 [bsz, vocab_size]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [bsz, vocab_size]

    # 创建移除掩码: 累积概率 > effective_top_p
    sorted_remove = cumulative_probs > effective_top_p  # 广播 [bsz, 1] -> [bsz, vocab_size]

    # 确保至少保留 min_tokens_to_keep 个 token
    min_keep_mask = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size) < min_tokens_to_keep
    sorted_remove = sorted_remove & (~min_keep_mask)

    # 转换到原始索引空间
    remove_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_remove)

    # 应用掩码
    probs.masked_fill_(remove_mask, 0.0)

    # ====== 5. 归一化 (处理零概率和) ======
    probs_sum = probs.sum(dim=-1, keepdim=True)
    # 安全除法: 防止零概率和 (理论上 min_tokens_to_keep 会避免此情况)
    probs = probs / torch.where(probs_sum > 0, probs_sum, torch.ones_like(probs_sum))

    # ====== 6. 采样 (完全并行) ======
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_tokens


def _sampler_mode() -> str:
    return os.environ.get("ALBATROSS_SAMPLER", "python").strip().lower()


def _sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1).to(torch.long)


def _sample_gumbel(logits: torch.Tensor, temperature: torch.Tensor, eps: float = 6.2e-5) -> torch.Tensor:
    scaled = logits / temperature
    noise = torch.empty_like(scaled).exponential_().clamp_(min=eps).log_().neg_()
    return torch.argmax(scaled + noise, dim=-1).to(torch.long)


def _cuda_sampler_available() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch.ops, "rwkv7_state_fwd_fp16")
        and hasattr(torch.ops.rwkv7_state_fwd_fp16, "setup_rand")
        and hasattr(torch.ops.rwkv7_state_fwd_fp16, "batch_sampling_temperature_topk_topp")
    )


def _all_rows_equal(tensor: torch.Tensor) -> bool:
    if tensor.shape[0] <= 1:
        return True
    return bool(torch.all(tensor == tensor[:1]).item())


def _cuda_uniform_parameters_available(
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    alpha_presence: torch.Tensor,
    alpha_frequency: torch.Tensor,
    penalty_decay: torch.Tensor,
) -> bool:
    return all(
        _all_rows_equal(tensor)
        for tensor in (
            temperature,
            top_p,
            top_k,
        )
    )


def _sample_cuda_uniform(
    logits: torch.Tensor,
    occurrence: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    alpha_presence: torch.Tensor,
    alpha_frequency: torch.Tensor,
    penalty_decay: torch.Tensor,
) -> torch.Tensor:
    logits_fp32 = logits.float().contiguous()
    batch = logits_fp32.shape[0]
    seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
    states = torch.ops.rwkv7_state_fwd_fp16.setup_rand(seed, batch)
    tokens = torch.ops.rwkv7_state_fwd_fp16.batch_sampling_temperature_topk_topp(
        logits_fp32,
        states,
        float(temperature[0, 0].item()),
        int(top_k[0, 0].item()),
        float(top_p[0, 0].item()),
    )
    return tokens.to(torch.long)


def sample_next_tokens_batch(
    *,
    logits: torch.Tensor,
    occurrence: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    alpha_presence: torch.Tensor,
    alpha_frequency: torch.Tensor,
    penalty_decay: torch.Tensor,
) -> torch.Tensor:
    mode = _sampler_mode()
    if mode == "python":
        return sample_logits_real_batch(logits, temperature, top_p, top_k).to(torch.long)
    if mode == "greedy":
        return _sample_greedy(logits)
    if mode == "gumbel":
        return _sample_gumbel(logits, temperature)
    if mode == "cuda":
        if _cuda_sampler_available() and _cuda_uniform_parameters_available(
            temperature,
            top_p,
            top_k,
            alpha_presence,
            alpha_frequency,
            penalty_decay,
        ):
            return _sample_cuda_uniform(
                logits,
                occurrence,
                temperature,
                top_p,
                top_k,
                alpha_presence,
                alpha_frequency,
                penalty_decay,
            )
        if os.environ.get("ALBATROSS_SAMPLER_FALLBACK", "1") == "1":
            return sample_logits_real_batch(logits, temperature, top_p, top_k).to(torch.long)
        raise RuntimeError(
            "ALBATROSS_SAMPLER=cuda requested but CUDA sampler is unavailable "
            "or active batch sampling parameters are not uniform"
        )
    raise ValueError(f"Unsupported ALBATROSS_SAMPLER={mode!r}")
