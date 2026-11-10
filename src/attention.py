"""Task 4: reference scaled dot-product attention and tiled attention with online softmax.

Shapes: q (B,H,Tq,D), k/v (B,H,Tk,D). Causal masking is aligned to the bottom-right
(query i may see key j iff j <= i + (Tk - Tq)), which is exactly what a KV-cache decode step needs.
"""
import math
import time

import torch
import torch.nn.functional as F


def reference_attention(q, k, v, causal=True, scale=None):
    Tq, Tk = q.size(-2), k.size(-2)
    scale = scale or 1.0 / math.sqrt(q.size(-1))
    s = (q.float() @ k.float().transpose(-2, -1)) * scale          # (B,H,Tq,Tk) materialised
    if causal:
        assert Tq <= Tk, "causal attention needs Tq <= Tk"
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        s = s.masked_fill(~mask, float("-inf"))
    s = s - s.amax(dim=-1, keepdim=True)                           # numerically stable softmax
    p = s.exp()
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal=True, scale=None, block_q=64, block_k=64):
    """FlashAttention-style forward: never materialises the (Tq,Tk) matrix.
    Per query tile keeps running max m, running normaliser l and running output acc:
        m' = max(m, rowmax(S));  alpha = exp(m - m');  P = exp(S - m')
        l  = l*alpha + rowsum(P);  acc = acc*alpha + P @ V       ->  out = acc / l
    """
    B, H, Tq, D = q.shape
    Tk = k.size(-2)
    scale = scale or 1.0 / math.sqrt(D)
    offset = Tk - Tq
    if causal:
        assert Tq <= Tk
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = qf[:, :, qs:qe]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros(B, H, qe - qs, 1, device=q.device)
        acc = torch.zeros(B, H, qe - qs, D, device=q.device)
        qi = torch.arange(qs, qe, device=q.device).unsqueeze(1)
        for ks in range(0, Tk, block_k):
            if causal and ks > (qe - 1) + offset:                  # whole tile is in the future -> skip
                break
            ke = min(ks + block_k, Tk)
            s = (qb @ kf[:, :, ks:ke].transpose(-2, -1)) * scale
            if causal:
                kj = torch.arange(ks, ke, device=q.device).unsqueeze(0)
                s = s.masked_fill(kj > qi + offset, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # fully-masked rows
            alpha = torch.exp(m - m_safe)
            p = torch.exp(s - m_safe)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + p @ vf[:, :, ks:ke]
            m = m_new
        out[:, :, qs:qe] = acc / l
    return out.to(q.dtype)


def sdpa_attention(q, k, v, causal=True):
    """PyTorch fused/optimised backend (for comparison only)."""
    if causal and q.size(-2) != k.size(-2):
        Tq, Tk = q.size(-2), k.size(-2)
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _timed(fn, device, repeats):
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return 1000 * sum(ts) / len(ts), peak


def benchmark(shapes=((1, 8, 128, 64), (1, 8, 512, 64), (1, 8, 1024, 64), (1, 8, 2048, 64)),
              device=None, dtype=torch.float32, repeats=5, block=64):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = []
    for (B, H, T, D) in shapes:
        g = torch.Generator(device="cpu").manual_seed(0)
        q, k, v = (torch.randn(B, H, T, D, generator=g).to(device, dtype) for _ in range(3))
        oracle = reference_attention(q.double(), k.double(), v.double())
        impls = {"reference": lambda: reference_attention(q, k, v),
                 "tiled_online_softmax": lambda: tiled_attention(q, k, v, block_q=block, block_k=block),
                 "torch_sdpa": lambda: sdpa_attention(q, k, v)}
        for name, fn in impls.items():
            out = fn()
            err = (out.double() - oracle).abs()
            ms, peak = _timed(fn, device, repeats)
            rows.append({"shape": f"{B}x{H}x{T}x{D}", "impl": name, "max_abs_err": err.max().item(),
                         "mean_abs_err": err.mean().item(), "ms": round(ms, 3), "peak_mem_mb": round(peak, 2),
                         "score_matrix_mb": round(B * H * T * T * 4 / 2**20, 2) if name == "reference" else
                                            round(B * H * block * block * 4 / 2**20, 4) if name.startswith("tiled") else None})
    return rows
