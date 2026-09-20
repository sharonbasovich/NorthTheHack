# Fused decode-step kernels for graph-node launch (compiled offline via
# nvrtc -> cubin/PTX, loaded by cuModuleLoadData, embedded as kernel nodes).
# All tensors bf16 except logits/accumulators (fp32). Layout mirrors
# engine.py _decode_step_slow exactly.

GSTEP_SRC = r"""
#include <cuda_bf16.h>
typedef __nv_bfloat16 bf16;
#define NT 256

// out[b][i] = x[b][i] * rsqrt(mean_b(x^2)+eps) * w[i]
extern "C" __global__ void rms_k(const bf16* __restrict__ x,
        const bf16* __restrict__ w, bf16* __restrict__ out,
        int H, int B, float eps) {
    int b = blockIdx.x;
    const bf16* xr = x + (long long)b * H;
    bf16* orr = out + (long long)b * H;
    __shared__ float red[NT / 32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += NT) {
        float v = __bfloat162float(xr[i]); acc += v * v;
    }
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < NT / 32) ? red[threadIdx.x] : 0.f;
        for (int off = 4; off; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / (float)H + eps);
    for (int i = threadIdx.x; i < H; i += NT)
        orr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv
                                  * __bfloat162float(w[i]));
}

// out[b][row] = sum_i x[b][i] * W[row][i];  W is [K,N] row-major.
// grid: ceil(K/8) * B blocks; 256 threads = 8 warps; warp per row.
extern "C" __global__ void gemv_k(const bf16* __restrict__ x,
        const bf16* __restrict__ W, bf16* __restrict__ out,
        int N, int K, int B) {
    int per = gridDim.x / B;
    int b = blockIdx.x / per;
    int row = (blockIdx.x - b * per) * 8 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= K) return;
    const bf16* xr = x + (long long)b * N;
    const bf16* wr = W + (long long)row * N;
    float acc = 0.f;
    for (int i = lane; i < N; i += 32)
        acc += __bfloat162float(xr[i]) * __bfloat162float(wr[i]);
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(long long)b * K + row] = __float2bfloat16(acc);
}

// out[b][row] += sum_i x[b][i] * W[row][i]  (residual epilogue)
extern "C" __global__ void gemv_add_k(const bf16* __restrict__ x,
        const bf16* __restrict__ W, bf16* __restrict__ out,
        int N, int K, int B) {
    int per = gridDim.x / B;
    int b = blockIdx.x / per;
    int row = (blockIdx.x - b * per) * 8 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= K) return;
    const bf16* xr = x + (long long)b * N;
    const bf16* wr = W + (long long)row * N;
    float acc = 0.f;
    for (int i = lane; i < N; i += 32)
        acc += __bfloat162float(xr[i]) * __bfloat162float(wr[i]);
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        out[(long long)b * K + row] = __float2bfloat16(
            __bfloat162float(out[(long long)b * K + row]) + acc);
}

// Per-head rms + half-split rope + KV cache write.
// grid: B*(NQ+NKV) blocks; block = 128 threads = D lanes.
// qkv row layout: [q(32h) | k(8h) | v(8h)] each D=128.
// k/v written to kc/vc[b][kvh][pos][d]; normed+roped q -> qe[b][qh][d].
extern "C" __global__ void rope_kv_k(bf16* __restrict__ qkv,
        const bf16* __restrict__ qn, const bf16* __restrict__ kn,
        const bf16* __restrict__ cost, const bf16* __restrict__ sint,
        const long long* __restrict__ pos,
        bf16* __restrict__ qe, bf16* __restrict__ kc,
        bf16* __restrict__ vc, int S, int NQ, int NKV, int D, float eps) {
    int nh = NQ + NKV;
    int b = blockIdx.x / nh;
    int h = blockIdx.x - b * nh;
    int d = threadIdx.x;
    long long p = pos[b];
    if (d >= D) return;
    bf16* src = qkv + ((long long)b * (NQ + 2 * NKV) + h) * D;
    const bf16* nw = (h < NQ) ? qn : kn;
    __shared__ float red[4];
    float v = __bfloat162float(src[d]);
    float acc = v * v;
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((d & 31) == 0) red[d >> 5] = acc;
    __syncthreads();
    if (d == 0) {
        float s = red[0] + red[1] + red[2] + red[3];
        red[0] = rsqrtf(s / (float)D + eps);
    }
    __syncthreads();
    float inv = red[0];
    float xn = v * inv * __bfloat162float(nw[d]);
    int half = D >> 1;
    bf16* other = (d < half) ? src + half + d : src - half + d;
    float vo = __bfloat162float(other[0]);
    float xo = vo * inv * __bfloat162float(
        nw[(d < half) ? d + half : d - half]);
    float cs = __bfloat162float(cost[p * D + d]);
    float sn = __bfloat162float(sint[p * D + d]);
    float r = (d < half) ? (xn * cs - xo * sn) : (xn * cs + xo * sn);
    bf16 val = __float2bfloat16(r);
    if (h < NQ) {
        qe[((long long)b * NQ + h) * D + d] = val;
    } else {
        int kv = h - NQ;
        kc[(((long long)b * NKV + kv) * S + p) * D + d] = val;
        // v lives in the separate tail block (NQ+NKV+kv), unnormed
        vc[(((long long)b * NKV + kv) * S + p) * D + d] =
            qkv[((long long)b * (NQ + 2 * NKV) + NQ + NKV + kv) * D + d];
    }
}

// GQA decode attention. grid: B*NKV blocks; block 128.
// Each block handles GROUP q-heads sharing one kv head over L=pos+1 slots.
// Warp 0-3 -> q head 0-3 of the group; within a warp, lanes stride s.
extern "C" __global__ void attn_k(const bf16* __restrict__ qe,
        const bf16* __restrict__ kc, const bf16* __restrict__ vc,
        const long long* __restrict__ pos,
        bf16* __restrict__ out,
        int S, int NKV, int GROUP, int D, int B, float scale) {
    int b = blockIdx.x / NKV;
    int kv = blockIdx.x - b * NKV;
    int L = (int)pos[b] + 1;
    int warp = threadIdx.x >> 5;         // q head within group (0..3)
    int lane = threadIdx.x & 31;
    if (warp >= GROUP) return;
    const bf16* q = qe + (((long long)b * NKV * GROUP) + kv * GROUP
                        + warp) * D;
    const bf16* kbase = kc + (((long long)b * NKV + kv) * S) * D;
    const bf16* vbase = vc + (((long long)b * NKV + kv) * S) * D;
    // q in registers
    float qf[128 / 32];                  // D/32 per lane
    for (int i = 0; i < D / 32; ++i)
        qf[i] = __bfloat162float(q[lane + i * 32]);
    // pass 1: max score
    float mx = -1e30f;
    for (int s = lane; s < L; s += 32) {
        const bf16* k = kbase + (long long)s * D;
        float acc = 0.f;
        for (int i = 0; i < D / 32; ++i)
            acc += qf[i] * __bfloat162float(k[lane + i * 32]);
        // cross-lane dot
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (acc * scale > mx) mx = acc * scale;
    }
    // pass 2: sum exp + weighted v
    float den = 0.f;
    float oacc[128 / 32];
    for (int i = 0; i < D / 32; ++i) oacc[i] = 0.f;
    for (int s = lane; s < L; s += 32) {
        const bf16* k = kbase + (long long)s * D;
        float acc = 0.f;
        for (int i = 0; i < D / 32; ++i)
            acc += qf[i] * __bfloat162float(k[lane + i * 32]);
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        float e = expf(acc * scale - mx);
        den += e;
        const bf16* vv = vbase + (long long)s * D;
        for (int i = 0; i < D / 32; ++i)
            oacc[i] += e * __bfloat162float(vv[lane + i * 32]);
    }
    for (int off = 16; off; off >>= 1)
        den += __shfl_xor_sync(0xffffffffu, den, off);
    float inv = 1.f / den;
    bf16* orow = out + (((long long)b * NKV * GROUP) + kv * GROUP
                        + warp) * D;
    for (int i = 0; i < D / 32; ++i)
        orow[lane + i * 32] = __float2bfloat16(oacc[i] * inv);
}

// m = silu(gu[:, :I]) * gu[:, I:]  (elementwise, I=9728)
extern "C" __global__ void silu_k(const bf16* __restrict__ gu,
        bf16* __restrict__ m, int I, int B) {
    int idx = blockIdx.x * NT + threadIdx.x;
    int tot = B * I;
    if (idx >= tot) return;
    int b = idx / I, i = idx - b * I;
    float a = __bfloat162float(gu[(long long)b * 2 * I + i]);
    float g = __bfloat162float(gu[(long long)b * 2 * I + I + i]);
    float s = a / (1.f + expf(-a));
    m[idx] = __float2bfloat16(s * g);
}

// argmax over [B, V] fp32 logits -> tok[b]. grid: B blocks.
extern "C" __global__ void argmax_k(const float* __restrict__ logits,
        long long* __restrict__ tok, int V) {
    int b = blockIdx.x;
    const float* r = logits + (long long)b * V;
    __shared__ float sv[NT / 32];
    __shared__ int si[NT / 32];
    float bv = -1e30f; int bi = -1;
    for (int i = threadIdx.x; i < V; i += NT) {
        float v = r[i];
        if (v > bv) { bv = v; bi = i; }
    }
    for (int off = 16; off; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv;
                                   si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        bv = sv[0]; bi = si[0];
        for (int i = 1; i < NT / 32; ++i)
            if (sv[i] > bv) { bv = sv[i]; bi = si[i]; }
        tok[b] = bi;
    }
}

// x[b][i] = emb[cur[b]][i]
extern "C" __global__ void embed_k(const bf16* __restrict__ emb,
        const long long* __restrict__ cur, bf16* __restrict__ x,
        int H, int B) {
    int idx = blockIdx.x * NT + threadIdx.x;
    int b = idx / H;
    if (b >= B) return;
    x[idx] = emb[cur[b] * (long long)H + (idx - b * H)];
}

// argmax over bf16 logits [B,V] -> cur[b]; also pos[b]++.
extern "C" __global__ void argmax_pos_k(const bf16* __restrict__ logits,
        long long* __restrict__ cur, long long* __restrict__ pos,
        int V, int B) {
    int b = blockIdx.x;
    const bf16* r = logits + (long long)b * V;
    __shared__ float sv[NT / 32];
    __shared__ int si[NT / 32];
    float bv = -1e30f; int bi = -1;
    for (int i = threadIdx.x; i < V; i += NT) {
        float v = __bfloat162float(r[i]);
        if (v > bv) { bv = v; bi = i; }
    }
    for (int off = 16; off; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv;
                                   si[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        bv = sv[0]; bi = si[0];
        for (int i = 1; i < NT / 32; ++i)
            if (sv[i] > bv) { bv = sv[i]; bi = si[i]; }
        cur[b] = bi;
        pos[b] += 1;
    }
}
"""
