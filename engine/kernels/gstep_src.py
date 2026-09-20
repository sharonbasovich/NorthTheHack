# Fused decode-step kernels for graph-node launch (compiled offline via
# nvrtc -> PTX, loaded by cuModuleLoadData, embedded as kernel nodes).
# All tensors bf16 except accumulators (fp32). Layout mirrors
# engine.py _decode_step_slow exactly. Vectorized: W rows read as uint4
# (8 bf16 lanes), warp-per-output-row.

GSTEP_SRC = r"""
#include <cuda_bf16.h>
typedef __nv_bfloat16 bf16;
#define NT 256

// x[b][i] = emb[cur[b]][i]
extern "C" __global__ void embed_k(const bf16* __restrict__ emb,
        const long long* __restrict__ cur, bf16* __restrict__ x,
        int H, int B) {
    int idx = blockIdx.x * NT + threadIdx.x;
    int b = idx / H;
    if (b >= B) return;
    x[idx] = emb[cur[b] * (long long)H + (idx - b * H)];
}

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

// out[b][row] = sum_i x[b][i] * W[row][i];  W [K,N] row-major, N%8==0.
// grid: ceil(K/8)*B blocks; 256 thr = 8 warps; warp per row, uint4 loads.
extern "C" __global__ void gemv_k(const bf16* __restrict__ x,
        const bf16* __restrict__ W, bf16* __restrict__ out,
        int N, int K, int B) {
    int per = gridDim.x / B;
    int b = blockIdx.x / per;
    int row = (blockIdx.x - b * per) * 8 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= K) return;
    int N8 = N >> 3;
    const uint4* xr = (const uint4*)(x + (long long)b * N);
    const uint4* wr = (const uint4*)(W + (long long)row * N);
    float acc = 0.f;
    for (int i = lane; i < N8; i += 32) {
        uint4 a = xr[i], wv = wr[i];
        const bf16* ab = (const bf16*)&a;
        const bf16* wb = (const bf16*)&wv;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += __bfloat162float(ab[j]) * __bfloat162float(wb[j]);
    }
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
    int N8 = N >> 3;
    const uint4* xr = (const uint4*)(x + (long long)b * N);
    const uint4* wr = (const uint4*)(W + (long long)row * N);
    float acc = 0.f;
    for (int i = lane; i < N8; i += 32) {
        uint4 a = xr[i], wv = wr[i];
        const bf16* ab = (const bf16*)&a;
        const bf16* wb = (const bf16*)&wv;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += __bfloat162float(ab[j]) * __bfloat162float(wb[j]);
    }
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
        out[(long long)b * K + row] = __float2bfloat16(
            __bfloat162float(out[(long long)b * K + row]) + acc);
}

// out[b][r] = silu(dot(x[b], W[r])) * dot(x[b], W[r+I]);  W [2I,N].
// grid: ceil(I/8)*B blocks; warp per row-pair.
extern "C" __global__ void gemv_silu_k(const bf16* __restrict__ x,
        const bf16* __restrict__ W, bf16* __restrict__ out,
        int N, int I, int B) {
    int per = gridDim.x / B;
    int b = blockIdx.x / per;
    int row = (blockIdx.x - b * per) * 8 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= I) return;
    int N8 = N >> 3;
    const uint4* xr = (const uint4*)(x + (long long)b * N);
    const uint4* w1 = (const uint4*)(W + (long long)row * N);
    const uint4* w2 = (const uint4*)(W + ((long long)row + I) * N);
    float a1 = 0.f, a2 = 0.f;
    for (int i = lane; i < N8; i += 32) {
        uint4 a = xr[i], v1 = w1[i], v2 = w2[i];
        const bf16* ab = (const bf16*)&a;
        const bf16* p1 = (const bf16*)&v1;
        const bf16* p2 = (const bf16*)&v2;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float xi = __bfloat162float(ab[j]);
            a1 += xi * __bfloat162float(p1[j]);
            a2 += xi * __bfloat162float(p2[j]);
        }
    }
    for (int off = 16; off; off >>= 1) {
        a1 += __shfl_down_sync(0xffffffffu, a1, off);
        a2 += __shfl_down_sync(0xffffffffu, a2, off);
    }
    if (lane == 0) {
        float s = a1 / (1.f + expf(-a1));
        out[(long long)b * I + row] = __float2bfloat16(s * a2);
    }
}

// Per-head rms + half-split rope + KV cache write.
// grid: B*(NQ+NKV) blocks; block = 128 threads = D lanes.
// qkv row layout: [q(32h) | k(8h) | v(8h)] each D=128.
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
        vc[(((long long)b * NKV + kv) * S + p) * D + d] =
            qkv[((long long)b * (NQ + 2 * NKV) + NQ + NKV + kv) * D + d];
    }
}

// GQA decode attention. grid: B*NKV blocks; block 128 = 4 warps,
// warp w handles q-head w of the group; lanes stride sequence.
extern "C" __global__ void attn_k(const bf16* __restrict__ qe,
        const bf16* __restrict__ kc, const bf16* __restrict__ vc,
        const long long* __restrict__ pos,
        bf16* __restrict__ out,
        int S, int NKV, int GROUP, int D, int B, float scale) {
    int b = blockIdx.x / NKV;
    int kv = blockIdx.x - b * NKV;
    int L = (int)pos[b] + 1;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= GROUP) return;
    const bf16* q = qe + (((long long)b * NKV * GROUP) + kv * GROUP
                        + warp) * D;
    const bf16* kbase = kc + (((long long)b * NKV + kv) * S) * D;
    const bf16* vbase = vc + (((long long)b * NKV + kv) * S) * D;
    float qf[4];
    for (int i = 0; i < 4; ++i)
        qf[i] = __bfloat162float(q[lane + i * 32]);
    float mx = -1e30f;
    for (int s = lane; s < L; s += 32) {
        const bf16* k = kbase + (long long)s * D;
        float acc = 0.f;
        for (int i = 0; i < 4; ++i)
            acc += qf[i] * __bfloat162float(k[lane + i * 32]);
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (acc * scale > mx) mx = acc * scale;
    }
    float den = 0.f;
    float oacc[4];
    for (int i = 0; i < 4; ++i) oacc[i] = 0.f;
    for (int s = lane; s < L; s += 32) {
        const bf16* k = kbase + (long long)s * D;
        float acc = 0.f;
        for (int i = 0; i < 4; ++i)
            acc += qf[i] * __bfloat162float(k[lane + i * 32]);
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        float e = expf(acc * scale - mx);
        den += e;
        const bf16* vv = vbase + (long long)s * D;
        for (int i = 0; i < 4; ++i)
            oacc[i] += e * __bfloat162float(vv[lane + i * 32]);
    }
    for (int off = 16; off; off >>= 1)
        den += __shfl_xor_sync(0xffffffffu, den, off);
    float inv = 1.f / den;
    bf16* orow = out + (((long long)b * NKV * GROUP) + kv * GROUP
                        + warp) * D;
    for (int i = 0; i < 4; ++i)
        orow[lane + i * 32] = __float2bfloat16(oacc[i] * inv);
}

// cur[b] = argmax(logits[b]); pos[b] += 1. logits bf16 [B,V].
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
