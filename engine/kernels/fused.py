"""Triton kernels for the fused decode step.

Arithmetic mirrors the reference exactly:
- RMSNorm reduces in fp32, casts the normalized value to bf16, then multiplies
  by the weight (Qwen3RMSNorm's order);
- RoPE multiplies bf16 tensors with the model's own cos/sin tables;
- attention is a flash-decode single pass with fp32 online softmax, grouped so
  each program handles one KV head and its 4 query heads (K/V read once);
- device-side `pos` drives the RoPE row, the cache slot, and the attention
  length, so a captured graph replays correctly as pos advances.
"""

import torch
import triton
import triton.language as tl

NEG_INF = float("-inf")


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    normed = (x * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + off, normed * w, mask=mask)


def rmsnorm(x, w, out, eps):
    _rmsnorm_kernel[(x.shape[0],)](x, w, out, x.shape[-1], eps,
                                  BLOCK=4096, num_warps=8)


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr, d_ptr, out_ptr, w_ptr, n_cols, eps, BLOCK: tl.constexpr
):
    # x <- x + d (bf16 residual add); out <- w * rmsnorm(x_new)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + off, mask=mask, other=0.0).to(tl.float32)
    sb = (x + d).to(tl.bfloat16)
    tl.store(x_ptr + off, sb, mask=mask)
    sf = sb.to(tl.float32)
    var = tl.sum(sf * sf, axis=0) / n_cols
    normed = (sf * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + off, normed * w, mask=mask)


def add_rmsnorm(x, delta, out, w, eps):
    """x += delta in bf16; out = rmsnorm(new x)."""
    _add_rmsnorm_kernel[(x.shape[0],)](x, delta, out, w, x.shape[-1], eps,
                                      BLOCK=4096, num_warps=8)


@triton.jit
def _qknorm_rope_cache_kernel(
    qkv_ptr, qnw_ptr, knw_ptr, cos_ptr, sin_ptr,
    kc_ptr, vc_ptr, qout_ptr, pos_ptr,
    stride_qkv, stride_kc_b, stride_kc_g, stride_qo_b,
    eps,
    Q_OFF: tl.constexpr, K_OFF: tl.constexpr, V_OFF: tl.constexpr,
    GROUP: tl.constexpr, D: tl.constexpr,
):
    g = tl.program_id(0)   # kv head
    b = tl.program_id(1)   # batch row
    pos = tl.load(pos_ptr)
    cols = tl.arange(0, D)
    half = cols < D // 2
    row = qkv_ptr + b * stride_qkv

    cos = tl.load(cos_ptr + pos * D + cols)
    sin = tl.load(sin_ptr + pos * D + cols)

    # ---- k head ----
    ksrc = row + K_OFF + g * D + cols
    kraw = tl.load(ksrc).to(tl.float32)
    kvar = tl.sum(kraw * kraw, axis=0) / D
    kinv = tl.math.rsqrt(kvar + eps)
    kn = (kraw * kinv).to(tl.bfloat16)
    # rotate_half(kn): lane c<64 takes -kn[c+64], c>=64 takes kn[c-64].
    # bf16(k_partner * inv) equals the already-rounded normed partner exactly.
    kpart = tl.load(ksrc + tl.where(half, D // 2, -D // 2)).to(tl.float32)
    knp = (kpart * kinv).to(tl.bfloat16)
    krot = tl.where(half, -knp, knp)
    ke = (kn * cos + krot * sin).to(tl.bfloat16)
    tl.store(kc_ptr + b * stride_kc_b + g * stride_kc_g + pos * D + cols, ke)

    # ---- v head: plain copy ----
    v = tl.load(row + V_OFF + g * D + cols)
    tl.store(vc_ptr + b * stride_kc_b + g * stride_kc_g + pos * D + cols, v)

    # ---- q heads g*GROUP .. g*GROUP+GROUP-1 ----
    for j in tl.static_range(GROUP):
        qsrc = row + Q_OFF + (g * GROUP + j) * D + cols
        qraw = tl.load(qsrc).to(tl.float32)
        qvar = tl.sum(qraw * qraw, axis=0) / D
        qinv = tl.math.rsqrt(qvar + eps)
        qn = (qraw * qinv).to(tl.bfloat16)
        qpart = tl.load(qsrc + tl.where(half, D // 2, -D // 2)).to(tl.float32)
        qnp = (qpart * qinv).to(tl.bfloat16)
        qrot = tl.where(half, -qnp, qnp)
        qe = (qn * cos + qrot * sin).to(tl.bfloat16)
        tl.store(qout_ptr + b * stride_qo_b + (g * GROUP + j) * D + cols, qe)


def qknorm_rope_cache(qkv, qnw, knw, cos, sin, kc, vc, qout, pos,
                      eps, nq, nkv, d):
    B = qkv.shape[0]
    _qknorm_rope_cache_kernel[(nkv, B)](
        qkv, qnw, knw, cos, sin, kc, vc, qout, pos,
        qkv.stride(0), kc.stride(0), kc.stride(1), qout.stride(0),
        eps,
        Q_OFF=0, K_OFF=nq * d, V_OFF=nq * d + nkv * d,
        GROUP=nq // nkv, D=d,
        num_warps=4,
    )


@triton.jit
def _attn_decode_kernel(
    q_ptr, kc_ptr, vc_ptr, out_ptr, pos_ptr,
    stride_q_b, stride_kc_b, stride_kc_g, stride_o_b,
    scale, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, PADM: tl.constexpr,
):
    g = tl.program_id(0)
    b = tl.program_id(1)
    pos = tl.load(pos_ptr)
    length = pos + 1

    cols = tl.arange(0, D)
    rows = tl.arange(0, PADM)
    rmask = rows < GROUP
    qb = q_ptr + b * stride_q_b + g * GROUP * D
    q = tl.load(qb + rows[:, None] * D + cols[None, :],
                mask=rmask[:, None], other=0.0)  # [PADM, D] bf16

    m_i = tl.full((PADM,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((PADM,), dtype=tl.float32)
    acc = tl.zeros((PADM, D), dtype=tl.float32)

    kbase = kc_ptr + b * stride_kc_b + g * stride_kc_g
    vbase = vc_ptr + b * stride_kc_b + g * stride_kc_g

    for s0 in range(0, length, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        smask = s < length
        k = tl.load(kbase + s[:, None] * D + cols[None, :],
                    mask=smask[:, None], other=0.0)  # [BLOCK_S, D] bf16
        sc = tl.dot(q, tl.trans(k)) * scale  # [PADM, BLOCK_S] fp32
        sc = tl.where(smask[None, :], sc, NEG_INF)
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])  # fp32
        v = tl.load(vbase + s[:, None] * D + cols[None, :],
                    mask=smask[:, None], other=0.0)  # [BLOCK_S, D] bf16
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    ob = out_ptr + b * stride_o_b + g * GROUP * D
    tl.store(ob + rows[:, None] * D + cols[None, :],
             out.to(tl.bfloat16), mask=rmask[:, None])


def attn_decode(q, kc, vc, out, pos, nkv, group, d, scale, block_s=128):
    padm = max(16, triton.next_power_of_2(group))
    _attn_decode_kernel[(nkv, q.shape[0])](
        q, kc, vc, out, pos,
        q.stride(0), kc.stride(0), kc.stride(1), out.stride(0),
        scale, GROUP=group, D=d, BLOCK_S=block_s, PADM=padm,
        num_warps=4,
    )


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, n_cols, I: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pid = tl.program_id(1)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(gu_ptr + row * (2 * I) + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * (2 * I) + I + cols, mask=mask, other=0.0).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    r = (sg.to(tl.float32) * u).to(tl.bfloat16)
    tl.store(out_ptr + row * n_cols + cols, r, mask=mask)


def silu_mul(gu, out, i_size):
    B = gu.shape[0]
    n = out.shape[-1]
    BLOCK = 1024
    _silu_mul_kernel[(B, triton.cdiv(n, BLOCK))](gu, out, n, I=i_size,
                                               BLOCK=BLOCK, num_warps=4)
