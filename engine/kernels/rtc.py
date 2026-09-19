"""Runtime-compiled fused decode kernels via NVRTC + CUDA driver API.

The eval box has no Triton, no nvcc, and gVisor makes CUDA graphs worthless
(nvproxy replays captured nodes through the same trapped-ioctl path). What
it DOES have: libnvrtc.so bundled inside the torch wheel and libcuda via
the driver. ctypes -> nvrtcCompileProgram -> cuModuleLoadData ->
cuLaunchKernel gives real fused kernels (~9 launches/layer vs ~30).

NOTE: cuda_bf16.h is NOT in nvrtc's include path on the eval box, so no
headers are used at all — bf16 is handled as unsigned-short bit patterns
with manual convert helpers (fp32 math + round-to-nearest-even, identical
to __hmul/__hadd semantics).

Numerics contract (matches the reference decode step's cast placement):
  - RMSNorm: fp32 accumulate of x^2 mean, rsqrt fp32, cast bf16, *w bf16.
  - rope: bf16 muls/adds.
  - scores: fp32 dot of bf16 q,k, *scale, -inf above pos, fp32 softmax,
    cast p to bf16, fp32 accumulate of p_bf16*v_bf16.
"""
import ctypes
import glob
import os
import torch

BF16_HELPERS = r"""
typedef unsigned short bf16;
__device__ __forceinline__ float bf2f(bf16 b) {
    return __uint_as_float(((unsigned)b) << 16);
}
__device__ __forceinline__ bf16 f2bf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);   // round-to-nearest-even
    return (bf16)(u >> 16);
}
__device__ __forceinline__ bf16 bfmul(bf16 a, bf16 b) {
    return f2bf(bf2f(a) * bf2f(b));
}
__device__ __forceinline__ bf16 bfadd(bf16 a, bf16 b) {
    return f2bf(bf2f(a) + bf2f(b));
}
__device__ __forceinline__ bf16 bfneg(bf16 a) { return a ^ 0x8000u; }
"""

RMS_SRC = BF16_HELPERS + r"""
extern "C" __global__ void rms_k(const bf16* __restrict__ x,
                                 const bf16* __restrict__ w,
                                 bf16* __restrict__ out,
                                 int n, float eps) {
    int row = blockIdx.x;
    const bf16* xr = x + (long long)row * n;
    bf16* orow = out + (long long)row * n;
    __shared__ float ssum[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bf2f(xr[i]);
        acc += v * v;
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = 0.f;
    int nw = (blockDim.x + 31) >> 5;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) tot += ssum[i];
    __shared__ float stot;
    if (threadIdx.x == 0) stot = rsqrtf(tot / (float)n + eps);
    __syncthreads();
    float r = stot;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        orow[i] = bfmul(w[i], f2bf(bf2f(xr[i]) * r));
}
"""

ROPE_CACHE_SRC = BF16_HELPERS + r"""
// grid.x = B * 48 (heads 0..31 q, 32..39 k, 40..47 v), block = 128 threads
extern "C" __global__ void rope_cache_k(
        const bf16* __restrict__ qkv,    // [B, 48*128]
        const bf16* __restrict__ qn,     // [128]
        const bf16* __restrict__ kn,     // [128]
        const bf16* __restrict__ cosb,   // [B, 128]
        const bf16* __restrict__ sinb,   // [B, 128]
        bf16* __restrict__ qbuf,         // [B, 32*128]
        bf16* __restrict__ kc,           // flat [B*8*cap,128]
        bf16* __restrict__ vc,
        const long long* __restrict__ pos, // [B]
        long long cap, float eps) {
    int b = blockIdx.x / 48;
    int h = blockIdx.x % 48;
    int d = threadIdx.x;
    const bf16* src = qkv + (long long)b * 48 * 128 + h * 128;
    if (h >= 40) {
        long long j = h - 40;
        long long idx = ((long long)b * 8 + j) * cap + pos[b];
        vc[idx * 128 + d] = src[d];
        return;
    }
    const bf16* w = (h < 32) ? qn : kn;
    __shared__ float ssum[4];
    float xv = bf2f(src[d]);
    float acc = xv * xv;
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = ssum[0] + ssum[1] + ssum[2] + ssum[3];
    __shared__ bf16 srope[128];
    float r = rsqrtf(tot / 128.f + eps);
    srope[d] = bfmul(w[d], f2bf(xv * r));
    __syncthreads();
    bf16 me = srope[d];
    bf16 rot = (d < 64) ? bfneg(srope[d + 64]) : srope[d - 64];
    bf16 ob = bfadd(bfmul(me, cosb[b * 128 + d]),
                    bfmul(rot, sinb[b * 128 + d]));
    if (h < 32) {
        qbuf[((long long)b * 32 + h) * 128 + d] = ob;
    } else {
        long long j = h - 32;
        long long idx = ((long long)b * 8 + j) * cap + pos[b];
        kc[idx * 128 + d] = ob;
    }
}
"""

ATTN_SRC = BF16_HELPERS + r"""
// grid.x = B * 32 (one block per q head), block = 128 threads.
extern "C" __global__ void attn_k(
        const bf16* __restrict__ qbuf,   // [B, 32*128]
        const bf16* __restrict__ kc,     // flat [B*8*cap,128]
        const bf16* __restrict__ vc,
        const long long* __restrict__ pos, // [B]
        bf16* __restrict__ out,          // [B, 32*128]
        long long cap, float scale) {
    int b = blockIdx.x / 32;
    int h = blockIdx.x % 32;
    int j = h >> 2;
    long long p = pos[b];
    const bf16* q = qbuf + ((long long)b * 32 + h) * 128;
    const bf16* kbase = kc + ((long long)b * 8 + j) * cap * 128;
    const bf16* vbase = vc + ((long long)b * 8 + j) * cap * 128;
    __shared__ float qs[128];
    __shared__ float red[32];
    extern __shared__ float scores[];
    qs[threadIdx.x] = bf2f(q[threadIdx.x]);
    __syncthreads();
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x) {
        const bf16* krow = kbase + (long long)s * 128;
        float dot = 0.f;
        for (int d = 0; d < 128; ++d)
            dot += qs[d] * bf2f(krow[d]);
        scores[s] = dot * scale;
    }
    __syncthreads();
    float mx = -3.402823466e+38f;
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
        mx = fmaxf(mx, scores[s]);
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 16));
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 1));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __shared__ float smax;
    if (threadIdx.x == 0) smax = mx;
    __syncthreads();
    mx = smax;
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
        scores[s] = expf(scores[s] - mx);
    __syncthreads();
    float sm = 0.f;
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x) sm += scores[s];
    sm += __shfl_down_sync(0xffffffffu, sm, 16);
    sm += __shfl_down_sync(0xffffffffu, sm, 8);
    sm += __shfl_down_sync(0xffffffffu, sm, 4);
    sm += __shfl_down_sync(0xffffffffu, sm, 2);
    sm += __shfl_down_sync(0xffffffffu, sm, 1);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sm;
    __syncthreads();
    sm = red[0] + red[1] + red[2] + red[3];
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = sm;
    __syncthreads();
    sm = ssum;
    bf16* pbv = (bf16*)(scores + (p + 2));
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
        pbv[s] = f2bf(scores[s] / sm);
    __syncthreads();
    int d = threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s <= (int)p; ++s)
        acc += bf2f(pbv[s]) * bf2f(vbase[(long long)s * 128 + d]);
    out[((long long)b * 32 + h) * 128 + d] = f2bf(acc);
}
"""

ATTN_MEGA_SRC = BF16_HELPERS + r"""
// One kernel for rope + cache write + attention. grid.x = B * 8 (one block
// per (b,j) kv head), block = 128 threads. Each block computes its own
// k and v cache entries inline, so there is no cross-block dependency.
// For each of its G=4 q heads: qk-norm, rope, fp32-dot softmax, bf16 pv.
extern "C" __global__ void attn_mega_k(
        const bf16* __restrict__ qkv,    // [B, 48*128]
        const bf16* __restrict__ qn,     // [128]
        const bf16* __restrict__ kn,     // [128]
        const bf16* __restrict__ cosb,   // [B, 128]
        const bf16* __restrict__ sinb,   // [B, 128]
        bf16* __restrict__ kc,           // flat [B*8*cap,128]
        bf16* __restrict__ vc,
        const long long* __restrict__ pos, // [B]
        bf16* __restrict__ out,          // [B, 32*128]
        long long cap, float scale, float eps) {
    int b = blockIdx.x / 8;
    int j = blockIdx.x % 8;
    int d = threadIdx.x;
    long long p = pos[b];
    const bf16* row = qkv + (long long)b * 48 * 128;
    __shared__ float red[32];
    __shared__ bf16 srope[128];

    // ---- write this block's k and v cache entries (needed by its own
    // attention below AND by later steps) ----
    {
        // v: raw copy, no norm/rope
        const bf16* vsrc = row + (40 + j) * 128;
        bf16 v = vsrc[d];
        long long idx = ((long long)b * 8 + j) * cap + p;
        vc[idx * 128 + d] = v;
        // k: kn norm then rope
        const bf16* ksrc = row + (32 + j) * 128;
        float xv = bf2f(ksrc[d]);
        float acc = xv * xv;
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        float tot = red[0] + red[1] + red[2] + red[3];
        float r = rsqrtf(tot / 128.f + eps);
        bf16 t = bfmul(kn[d], f2bf(xv * r));
        srope[d] = t;
        __syncthreads();
        bf16 rot = (d < 64) ? bfneg(srope[d + 64]) : srope[d - 64];
        bf16 ke = bfadd(bfmul(t, cosb[b * 128 + d]),
                        bfmul(rot, sinb[b * 128 + d]));
        kc[idx * 128 + d] = ke;
    }
    __syncthreads();

    const bf16* kbase = kc + ((long long)b * 8 + j) * cap * 128;
    const bf16* vbase = vc + ((long long)b * 8 + j) * cap * 128;
    // scores buffer sized (cap+pad) fp32 + (cap+pad) bf16
    extern __shared__ float scores[];

    for (int g = 0; g < 4; ++g) {
        int h = j * 4 + g;
        // q norm + rope into srope
        const bf16* qsrc = row + h * 128;
        float xv = bf2f(qsrc[d]);
        float acc = xv * xv;
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        float tot = red[0] + red[1] + red[2] + red[3];
        float r = rsqrtf(tot / 128.f + eps);
        bf16 t = bfmul(qn[d], f2bf(xv * r));
        srope[d] = t;
        __syncthreads();
        bf16 rot = (d < 64) ? bfneg(srope[d + 64]) : srope[d - 64];
        bf16 qe = bfadd(bfmul(t, cosb[b * 128 + d]),
                        bfmul(rot, sinb[b * 128 + d]));
        srope[d] = qe;
        __syncthreads();
        // score every cache row s <= p (the just-written row at s == p was
        // committed to kc above, so kbase covers it)
        for (int s = threadIdx.x; s <= (int)p; s += blockDim.x) {
            const bf16* krow = kbase + (long long)s * 128;
            float dot = 0.f;
            for (int dd = 0; dd < 128; ++dd)
                dot += bf2f(srope[dd]) * bf2f(krow[dd]);
            scores[s] = dot * scale;
        }
        __syncthreads();
        float mx = -3.402823466e+38f;
        for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
            mx = fmaxf(mx, scores[s]);
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 16));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 8));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 4));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 1));
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
        __syncthreads();
        mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        __shared__ float smax;
        if (threadIdx.x == 0) smax = mx;
        __syncthreads();
        mx = smax;
        for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
            scores[s] = expf(scores[s] - mx);
        __syncthreads();
        float sm = 0.f;
        for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
            sm += scores[s];
        sm += __shfl_down_sync(0xffffffffu, sm, 16);
        sm += __shfl_down_sync(0xffffffffu, sm, 8);
        sm += __shfl_down_sync(0xffffffffu, sm, 4);
        sm += __shfl_down_sync(0xffffffffu, sm, 2);
        sm += __shfl_down_sync(0xffffffffu, sm, 1);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sm;
        __syncthreads();
        sm = red[0] + red[1] + red[2] + red[3];
        __shared__ float ssum;
        if (threadIdx.x == 0) ssum = sm;
        __syncthreads();
        sm = ssum;
        bf16* pbv = (bf16*)(scores + (p + 2));
        for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
            pbv[s] = f2bf(scores[s] / sm);
        __syncthreads();
        float accv = 0.f;
        for (int s = 0; s <= (int)p; ++s)
            accv += bf2f(pbv[s]) * bf2f(vbase[(long long)s * 128 + d]);
        out[((long long)b * 32 + h) * 128 + d] = f2bf(accv);
        __syncthreads();
    }
}
"""

SILU_SRC = BF16_HELPERS + r"""
extern "C" __global__ void silu_k(const bf16* __restrict__ gu,
                                  bf16* __restrict__ out, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / I, i = idx % I;
    float a = bf2f(gu[(long long)b * 2 * I + i]);
    float g = bf2f(gu[(long long)b * 2 * I + I + i]);
    float s = a / (1.f + expf(-a));
    out[idx] = bfmul(f2bf(s), f2bf(g));
}
"""

# Whole-step persistent kernel: one launch per decode step executes all 36
# layers with software grid barriers between stages. Grid must be fully
# co-resident: launched with nblk = SM count, 256 threads, modest smem —
# guaranteed at least one resident block per SM on any modern part.
MEGA_SRC = BF16_HELPERS + r"""
#ifdef COOP
extern "C" __device__ unsigned cudaCGGetIntrinsicHandle(unsigned long long*);
extern "C" __device__ void cudaCGSynchronizeGrid(unsigned long long);
#endif
#define NT 512
#define HDIM 2560
#define VDIM 151936
#define IDIM 9728
#define QKVD 4608
#define ODIM 4096
#define GDIM 19456

#ifdef COOP
__device__ __forceinline__ void gbar(unsigned* cnt, volatile unsigned* gen) {
    unsigned long long h;
    cudaCGGetIntrinsicHandle(&h);
    cudaCGSynchronizeGrid(h);
}
#else
__device__ __forceinline__ void gbar(unsigned* cnt, volatile unsigned* gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned g = *gen;
        if (atomicAdd(cnt, 1u) == gridDim.x - 1) {
            *cnt = 0;
            __threadfence();
            atomicExch((unsigned*)gen, g + 1);
        } else {
            while (*gen == g) { }
        }
    }
    __syncthreads();
}
#endif

__device__ __forceinline__ float prod8(const int4 wv, const int4 xv) {
    const unsigned* wu = (const unsigned*)&wv;
    const unsigned* xu = (const unsigned*)&xv;
    float acc = 0.f;
    for (int i = 0; i < 4; ++i) {
        acc += bf2f((bf16)(wu[i] & 0xffff)) * bf2f((bf16)(xu[i] & 0xffff));
        acc += bf2f((bf16)(wu[i] >> 16)) * bf2f((bf16)(xu[i] >> 16));
    }
    return acc;
}

__device__ __forceinline__ float dot8(const bf16* w, const bf16* x,
                                      int k8) {
    // 8 bf16 pairs starting at element k8*8 — two 16B loads
    const int4 wv = *(const int4*)(w + (long long)k8 * 8);
    const int4 xv = *(const int4*)(x + (long long)k8 * 8);
    return prod8(wv, xv);
}

// one block reduces one row: bf16 x in, rms -> bf16 out (w bf16-mul)
__device__ void rms_row(const bf16* x, const bf16* w, bf16* o,
                        float eps, float* red) {
    float acc = 0.f;
    for (int i = threadIdx.x; i < HDIM; i += NT) {
        float v = bf2f(x[i]);
        acc += v * v;
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < NT / 32; ++i) tot += red[i];
    float r = rsqrtf(tot / (float)HDIM + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < HDIM; i += NT)
        o[i] = bfmul(w[i], f2bf(bf2f(x[i]) * r));
}

// warp-per-output gemv over all rows: O[b,i] = sum_k W[i,k] * X[b,k]
// optional bf16 residual add into O, optional fused silu reading GU.
__device__ void gemv(const bf16* W, const bf16* X, bf16* O,
                     int B, int Od, int K, bf16* res,
                     const bf16* gu, int fused) {
    int gw = blockIdx.x * (NT / 32) + (threadIdx.x >> 5);
    int nw = gridDim.x * (NT / 32);
    int lane = threadIdx.x & 31;
    int nout = B * Od;
    if (!fused) {
        // two output rows per warp, software-pipelined: 4 loads in flight
        for (int o0 = gw * 2; o0 < nout; o0 += nw * 2) {
            int o1 = o0 + 1;
            const bf16* xr0 = X + (long long)(o0 / Od) * K;
            const bf16* xr1 = o1 < nout ? X + (long long)(o1 / Od) * K
                                        : xr0;
            const bf16* wr0 = W + (long long)(o0 % Od) * K;
            const bf16* wr1 = o1 < nout ? W + (long long)(o1 % Od) * K
                                        : wr0;
            float acc0 = 0.f, acc1 = 0.f;
            int K8 = K / 8, k8 = lane;
            int4 w0 = {}, x0 = {}, w1 = {}, x1 = {};
            if (k8 < K8) {
                w0 = *(const int4*)(wr0 + (long long)k8 * 8);
                x0 = *(const int4*)(xr0 + (long long)k8 * 8);
                w1 = *(const int4*)(wr1 + (long long)k8 * 8);
                x1 = *(const int4*)(xr1 + (long long)k8 * 8);
            }
            for (; k8 < K8; k8 += 32) {
                int k8n = k8 + 32;
                int4 w0n = {}, x0n = {}, w1n = {}, x1n = {};
                if (k8n < K8) {
                    w0n = *(const int4*)(wr0 + (long long)k8n * 8);
                    x0n = *(const int4*)(xr0 + (long long)k8n * 8);
                    w1n = *(const int4*)(wr1 + (long long)k8n * 8);
                    x1n = *(const int4*)(xr1 + (long long)k8n * 8);
                }
                acc0 += prod8(w0, x0); acc1 += prod8(w1, x1);
                w0 = w0n; x0 = x0n; w1 = w1n; x1 = x1n;
            }
            acc0 += __shfl_down_sync(0xffffffffu, acc0, 16);
            acc0 += __shfl_down_sync(0xffffffffu, acc0, 8);
            acc0 += __shfl_down_sync(0xffffffffu, acc0, 4);
            acc0 += __shfl_down_sync(0xffffffffu, acc0, 2);
            acc0 += __shfl_down_sync(0xffffffffu, acc0, 1);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, 16);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, 8);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, 4);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, 2);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, 1);
            if (lane == 0) {
                O[o0] = res ? bfadd(res[o0], f2bf(acc0)) : f2bf(acc0);
                if (o1 < nout)
                    O[o1] = res ? bfadd(res[o1], f2bf(acc1)) : f2bf(acc1);
            }
        }
        return;
    }
    for (int o = gw; o < nout; o += nw) {
        int b = o / Od, i = o % Od;
        const bf16* xr = gu + (long long)b * 2 * K;
        const bf16* wr = W + (long long)i * K;
        float acc = 0.f;
        int K8 = K / 8;
        for (int k8 = lane; k8 < K8; k8 += 32) {
            {
                // silu(x) * u on the fly, 8 lanes of work at once
                const int4 av = *(const int4*)(xr + (long long)k8 * 8);
                const int4 gv = *(const int4*)(xr + K + (long long)k8 * 8);
                const int4 wv = *(const int4*)(wr + (long long)k8 * 8);
                const unsigned* au = (const unsigned*)&av;
                const unsigned* gu2 = (const unsigned*)&gv;
                const unsigned* wu = (const unsigned*)&wv;
                for (int i = 0; i < 4; ++i) {
                    float a0 = bf2f((bf16)(au[i] & 0xffff));
                    float a1 = bf2f((bf16)(au[i] >> 16));
                    bf16 m0 = bfmul(f2bf(a0 / (1.f + expf(-a0))),
                                    f2bf(bf2f((bf16)(gu2[i] & 0xffff))));
                    bf16 m1 = bfmul(f2bf(a1 / (1.f + expf(-a1))),
                                    f2bf(bf2f((bf16)(gu2[i] >> 16))));
                    acc += bf2f((bf16)(wu[i] & 0xffff)) * bf2f(m0);
                    acc += bf2f((bf16)(wu[i] >> 16)) * bf2f(m1);
                }
            }
        }
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if (lane == 0) {
            if (res) O[o] = bfadd(res[o], f2bf(acc));
            else O[o] = f2bf(acc);
        }
    }
}

// group-local gemv: one sequence's row, warps gwl..gwl+nwl cover outputs.
// 4 output rows per warp, 2-deep prefetch → 8 weight loads in flight.
__device__ void gemv_g(const bf16* W, const bf16* xr, bf16* O,
                       int Od, int K, bf16* res, const bf16* gu,
                       int fused, int gw, int gwtot) {
    int lane = threadIdx.x & 31;
    int K8 = K / 8;
    if (!fused) {
        for (int o0 = gw * 4; o0 < Od; o0 += gwtot * 4) {
            const bf16* wr[4];
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int t = 0; t < 4; ++t) {
                int o = o0 + t;
                wr[t] = o < Od ? W + (long long)o * K : wr[0];
            }
            int k8 = lane;
            int4 wv[4], xv[4];
            if (k8 < K8) {
                for (int t = 0; t < 4; ++t) {
                    wv[t] = *(const int4*)(wr[t] + (long long)k8 * 8);
                    xv[t] = *(const int4*)(xr + (long long)k8 * 8);
                }
            }
            for (; k8 < K8; k8 += 32) {
                int k8n = k8 + 32;
                int4 wn[4], xn[4];
                if (k8n < K8)
                    for (int t = 0; t < 4; ++t) {
                        wn[t] = *(const int4*)(wr[t]
                                               + (long long)k8n * 8);
                        xn[t] = *(const int4*)(xr + (long long)k8n * 8);
                    }
                for (int t = 0; t < 4; ++t) acc[t] += prod8(wv[t], xv[t]);
                for (int t = 0; t < 4; ++t) { wv[t] = wn[t]; xv[t] = xn[t]; }
            }
            for (int t = 0; t < 4; ++t) {
                float a = acc[t];
                a += __shfl_down_sync(0xffffffffu, a, 16);
                a += __shfl_down_sync(0xffffffffu, a, 8);
                a += __shfl_down_sync(0xffffffffu, a, 4);
                a += __shfl_down_sync(0xffffffffu, a, 2);
                a += __shfl_down_sync(0xffffffffu, a, 1);
                int o = o0 + t;
                if (lane == 0 && o < Od)
                    O[o] = res ? bfadd(res[o], f2bf(a)) : f2bf(a);
            }
        }
        return;
    }
    for (int o = gw; o < Od; o += gwtot) {
        const bf16* wr = W + (long long)o * K;
        float acc = 0.f;
        for (int k8 = lane; k8 < K8; k8 += 32) {
            const int4 av = *(const int4*)(gu + (long long)k8 * 8);
            const int4 gv = *(const int4*)(gu + K + (long long)k8 * 8);
            const int4 wv = *(const int4*)(wr + (long long)k8 * 8);
            const unsigned* au = (const unsigned*)&av;
            const unsigned* gu2 = (const unsigned*)&gv;
            const unsigned* wu = (const unsigned*)&wv;
            for (int i = 0; i < 4; ++i) {
                float a0 = bf2f((bf16)(au[i] & 0xffff));
                float a1 = bf2f((bf16)(au[i] >> 16));
                bf16 m0 = bfmul(f2bf(a0 / (1.f + expf(-a0))),
                                f2bf(bf2f((bf16)(gu2[i] & 0xffff))));
                bf16 m1 = bfmul(f2bf(a1 / (1.f + expf(-a1))),
                                f2bf(bf2f((bf16)(gu2[i] >> 16))));
                acc += bf2f((bf16)(wu[i] & 0xffff)) * bf2f(m0);
                acc += bf2f((bf16)(wu[i] >> 16)) * bf2f(m1);
            }
        }
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if (lane == 0) O[o] = res ? bfadd(res[o], f2bf(acc)) : f2bf(acc);
    }
}

// group-local lm head: float logits out for one sequence.
// x row shared across outputs — prefetch weights only, 4 rows/warp.
__device__ void gemv_gf(const bf16* W, const bf16* xr, float* O,
                        int Od, int K, int gw, int gwtot) {
    int lane = threadIdx.x & 31;
    int K8 = K / 8;
    for (int o0 = gw * 4; o0 < Od; o0 += gwtot * 4) {
        const bf16* wr[4];
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int t = 0; t < 4; ++t) {
            int o = o0 + t;
            wr[t] = o < Od ? W + (long long)o * K : wr[0];
        }
        int k8 = lane;
        int4 wv[4], xv = {};
        if (k8 < K8) {
            xv = *(const int4*)(xr + (long long)k8 * 8);
            for (int t = 0; t < 4; ++t)
                wv[t] = *(const int4*)(wr[t] + (long long)k8 * 8);
        }
        for (; k8 < K8; k8 += 32) {
            int k8n = k8 + 32;
            int4 wn[4], xn = {};
            if (k8n < K8) {
                xn = *(const int4*)(xr + (long long)k8n * 8);
                for (int t = 0; t < 4; ++t)
                    wn[t] = *(const int4*)(wr[t] + (long long)k8n * 8);
            }
            for (int t = 0; t < 4; ++t) acc[t] += prod8(wv[t], xv);
            for (int t = 0; t < 4; ++t) wv[t] = wn[t];
            xv = xn;
        }
        for (int t = 0; t < 4; ++t) {
            float a = acc[t];
            a += __shfl_down_sync(0xffffffffu, a, 16);
            a += __shfl_down_sync(0xffffffffu, a, 8);
            a += __shfl_down_sync(0xffffffffu, a, 4);
            a += __shfl_down_sync(0xffffffffu, a, 2);
            a += __shfl_down_sync(0xffffffffu, a, 1);
            int o = o0 + t;
            if (lane == 0 && o < Od) O[o] = a;
        }
    }
}

// float-output variant for the lm head
__device__ void gemv_f(const bf16* W, const bf16* X, float* O,
                       int B, int Od, int K) {
    int gw = blockIdx.x * (NT / 32) + (threadIdx.x >> 5);
    int nw = gridDim.x * (NT / 32);
    int lane = threadIdx.x & 31;
    for (int o0 = gw * 2; o0 < B * Od; o0 += nw * 2) {
        int o1 = o0 + 1;
        const bf16* xr0 = X + (long long)(o0 / Od) * K;
        const bf16* xr1 = o1 < B * Od ? X + (long long)(o1 / Od) * K
                                      : xr0;
        const bf16* wr0 = W + (long long)(o0 % Od) * K;
        const bf16* wr1 = o1 < B * Od ? W + (long long)(o1 % Od) * K
                                      : wr0;
        float acc0 = 0.f, acc1 = 0.f;
        int K8 = K / 8, k8 = lane;
        int4 w0 = {}, x0 = {}, w1 = {}, x1 = {};
        if (k8 < K8) {
            w0 = *(const int4*)(wr0 + (long long)k8 * 8);
            x0 = *(const int4*)(xr0 + (long long)k8 * 8);
            w1 = *(const int4*)(wr1 + (long long)k8 * 8);
            x1 = *(const int4*)(xr1 + (long long)k8 * 8);
        }
        for (; k8 < K8; k8 += 32) {
            int k8n = k8 + 32;
            int4 w0n = {}, x0n = {}, w1n = {}, x1n = {};
            if (k8n < K8) {
                w0n = *(const int4*)(wr0 + (long long)k8n * 8);
                x0n = *(const int4*)(xr0 + (long long)k8n * 8);
                w1n = *(const int4*)(wr1 + (long long)k8n * 8);
                x1n = *(const int4*)(xr1 + (long long)k8n * 8);
            }
            acc0 += prod8(w0, x0); acc1 += prod8(w1, x1);
            w0 = w0n; x0 = x0n; w1 = w1n; x1 = x1n;
        }
        acc0 += __shfl_down_sync(0xffffffffu, acc0, 16);
        acc0 += __shfl_down_sync(0xffffffffu, acc0, 8);
        acc0 += __shfl_down_sync(0xffffffffu, acc0, 4);
        acc0 += __shfl_down_sync(0xffffffffu, acc0, 2);
        acc0 += __shfl_down_sync(0xffffffffu, acc0, 1);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, 16);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, 8);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, 4);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, 2);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, 1);
        if (lane == 0) {
            O[o0] = acc0;
            if (o1 < B * Od) O[o1] = acc1;
        }
    }
}

// rope+cache+attention for one (b,j) pair; executed by threads 0..127 of
// a block (warps 0-3). scores smem buffer provided by caller.
__device__ void attn_unit(int b, int j, const bf16* row,
                          const bf16* qn, const bf16* kn,
                          const bf16* cost, const bf16* sint,
                          bf16* kc, bf16* vc, long long p,
                          bf16* out, long long cap,
                          float scale, float eps,
                          float* red, bf16* srope, float* scores) {
    int d = threadIdx.x;   // < 128 guaranteed by caller guard
    if (d >= 128) return;
    const bf16* cosb = cost + p * 128;
    const bf16* sinb = sint + p * 128;
    // write k/v cache entries
    {
        const bf16* vsrc = row + (40 + j) * 128;
        long long idx = ((long long)b * 8 + j) * cap + p;
        vc[idx * 128 + d] = vsrc[d];
        const bf16* ksrc = row + (32 + j) * 128;
        float xv = bf2f(ksrc[d]);
        float acc = xv * xv;
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if ((d & 31) == 0) red[d >> 5] = acc;
        __syncthreads();
        float tot = red[0] + red[1] + red[2] + red[3];
        float r = rsqrtf(tot / 128.f + eps);
        bf16 t = bfmul(kn[d], f2bf(xv * r));
        srope[d] = t;
        __syncthreads();
        bf16 rot = (d < 64) ? bfneg(srope[d + 64]) : srope[d - 64];
        kc[idx * 128 + d] = bfadd(bfmul(t, cosb[d]),
                                  bfmul(rot, sinb[d]));
    }
    __syncthreads();
    const bf16* kbase = kc + ((long long)b * 8 + j) * cap * 128;
    const bf16* vbase = vc + ((long long)b * 8 + j) * cap * 128;
    for (int g = 0; g < 4; ++g) {
        int h = j * 4 + g;
        const bf16* qsrc = row + h * 128;
        float xv = bf2f(qsrc[d]);
        float acc = xv * xv;
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if ((d & 31) == 0) red[d >> 5] = acc;
        __syncthreads();
        float tot = red[0] + red[1] + red[2] + red[3];
        float r = rsqrtf(tot / 128.f + eps);
        bf16 t = bfmul(qn[d], f2bf(xv * r));
        srope[d] = t;
        __syncthreads();
        bf16 rot = (d < 64) ? bfneg(srope[d + 64]) : srope[d - 64];
        srope[d] = bfadd(bfmul(t, cosb[d]),
                         bfmul(rot, sinb[d]));
        __syncthreads();
        for (int s = d; s <= (int)p; s += 128) {
            const bf16* krow = kbase + (long long)s * 128;
            float dot = 0.f;
            for (int dd = 0; dd < 128; ++dd)
                dot += bf2f(srope[dd]) * bf2f(krow[dd]);
            scores[s] = dot * scale;
        }
        __syncthreads();
        float mx = -3.402823466e+38f;
        for (int s = d; s <= (int)p; s += 128) mx = fmaxf(mx, scores[s]);
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 16));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 8));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 4));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, 1));
        if ((d & 31) == 0) red[d >> 5] = mx;
        __syncthreads();
        mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        __syncthreads();
        for (int s = d; s <= (int)p; s += 128)
            scores[s] = expf(scores[s] - mx);
        __syncthreads();
        float sm = 0.f;
        for (int s = d; s <= (int)p; s += 128) sm += scores[s];
        sm += __shfl_down_sync(0xffffffffu, sm, 16);
        sm += __shfl_down_sync(0xffffffffu, sm, 8);
        sm += __shfl_down_sync(0xffffffffu, sm, 4);
        sm += __shfl_down_sync(0xffffffffu, sm, 2);
        sm += __shfl_down_sync(0xffffffffu, sm, 1);
        if ((d & 31) == 0) red[d >> 5] = sm;
        __syncthreads();
        sm = red[0] + red[1] + red[2] + red[3];
        __syncthreads();
        bf16* pbv = (bf16*)(scores + (p + 2));
        for (int s = d; s <= (int)p; s += 128)
            pbv[s] = f2bf(scores[s] / sm);
        __syncthreads();
        float accv = 0.f;
        for (int s = 0; s <= (int)p; ++s)
            accv += bf2f(pbv[s]) * bf2f(vbase[(long long)s * 128 + d]);
        out[((long long)b * 32 + h) * 128 + d] = f2bf(accv);
        __syncthreads();
    }
}

extern "C" __global__ void step_all_k(
        const long long* __restrict__ lw,   // [NL*10] weight+cache ptrs
        const bf16* __restrict__ embed,     // [V,HDIM]
        const bf16* __restrict__ finw,
        const bf16* __restrict__ cost,      // [CAPT,128]
        const bf16* __restrict__ sint,
        long long* __restrict__ pos,        // [B]
        long long* __restrict__ cur,        // [B]
        bf16* __restrict__ hid,             // [B,HDIM]
        bf16* __restrict__ hbuf,            // [B,2*HDIM]
        bf16* __restrict__ qkv,             // [B,QKVD]
        bf16* __restrict__ obuf,            // [B,ODIM]
        bf16* __restrict__ gu,              // [B,GDIM]
        float* __restrict__ logits,         // [B,VDIM]
        float* __restrict__ amaxv,          // [NB*B]
        int* __restrict__ amaxi,            // [NB*B]
        unsigned* __restrict__ cnt,
        unsigned* __restrict__ gen,
        volatile long long* __restrict__ flag,   // host-mapped
        volatile long long* __restrict__ tokm,   // host-mapped [O*B]
        int B, int NL, int ntok, float eps, long long cap, int bench) {
    int blk = blockIdx.x, tid = threadIdx.x;
    (void)tid;
    int nblk = gridDim.x;
    __shared__ float red[NT / 32];
    __shared__ bf16 srope[128];
    __shared__ float sval[NT / 32];
    __shared__ int sidx[NT / 32];
    extern __shared__ float scores[];

    bf16* h = hbuf;
    bf16* h2 = hbuf + (long long)B * HDIM;
    int per = nblk / B;
    // batch-row block groups: block blk owns row b = blk/per at slot
    // loc = blk%per. Rows never interact — barriers are group-local
    // (cnt[b]/gen[b]), shrinking each barrier from 131 to per blocks.
    int b = blk / per;
    int loc = blk % per;
    if (blk >= B * per) return;   // spare blocks sit out entirely

    unsigned* gcnt = cnt + b;
    volatile unsigned* ggen = gen + b;
    int nwl = per * (NT / 32);
    int gwl = loc * (NT / 32) + (tid >> 5);

    if (bench) {
        for (int s = 0; s < ntok; ++s) {
            for (int i = 0; i < NL * 6 + 4; ++i) gbar(gcnt, ggen);
            if (loc == 0 && tid == 0) flag[b] = (long long)(s + 1);
        }
        return;
    }

    bf16* hidb = hid + (long long)b * HDIM;
    bf16* qkvb = qkv + (long long)b * QKVD;
    bf16* obufb = obuf + (long long)b * ODIM;
    bf16* gub  = gu + (long long)b * 2 * IDIM;
    float* logb = logits + (long long)b * VDIM;

    for (int step_i = 0; step_i < ntok; ++step_i) {
    // embed current token into residual stream
    if (loc == 0) {
        long long tok = cur[b];
        const bf16* e = embed + tok * (long long)HDIM;
        for (int i = tid; i < HDIM; i += NT) hidb[i] = e[i];
    }
    gbar(gcnt, ggen);

    for (int l = 0; l < NL; ++l) {
        const bf16* w_ln_in = (const bf16*)lw[l * 10 + 0];
        const bf16* wqkv    = (const bf16*)lw[l * 10 + 1];
        const bf16* wqn     = (const bf16*)lw[l * 10 + 2];
        const bf16* wkn     = (const bf16*)lw[l * 10 + 3];
        const bf16* wo      = (const bf16*)lw[l * 10 + 4];
        const bf16* w_ln2   = (const bf16*)lw[l * 10 + 5];
        const bf16* wgu     = (const bf16*)lw[l * 10 + 6];
        const bf16* wd      = (const bf16*)lw[l * 10 + 7];
        bf16* kcb           = (bf16*)lw[l * 10 + 8];
        bf16* vcb           = (bf16*)lw[l * 10 + 9];

        if (loc == 0) rms_row(hidb, w_ln_in, h + (long long)b * HDIM,
                              eps, red);
        gbar(gcnt, ggen);
        gemv_g(wqkv, h + (long long)b * HDIM, qkvb, QKVD, HDIM, 0, 0,
               0, gwl, nwl);
        gbar(gcnt, ggen);
        if (loc < 8) {
            attn_unit(b, loc, qkvb, wqn, wkn,
                      cost, sint, kcb, vcb, pos[b],
                      obufb, cap, 1.0f / 11.313708499f, eps,
                      red, srope, scores);
        }
        gbar(gcnt, ggen);
        gemv_g(wo, obufb, hidb, HDIM, ODIM, hidb, 0, 0, gwl, nwl);
        gbar(gcnt, ggen);
        if (loc == 0) rms_row(hidb, w_ln2, h2 + (long long)b * HDIM,
                              eps, red);
        gbar(gcnt, ggen);
        gemv_g(wgu, h2 + (long long)b * HDIM, gub, GDIM, HDIM, 0, 0,
               0, gwl, nwl);
        gbar(gcnt, ggen);
        gemv_g(wd, 0, hidb, HDIM, IDIM, hidb, gub, 1, gwl, nwl);
        gbar(gcnt, ggen);
    }

    if (loc == 0) rms_row(hidb, finw, h + (long long)b * HDIM, eps,
                          red);
    gbar(gcnt, ggen);
    gemv_gf(embed, h + (long long)b * HDIM, logb, VDIM, HDIM,
            gwl, nwl);
    gbar(gcnt, ggen);

    // argmax over logits[b]: group-local slice scan, scratch reduce,
    // loc==0 reduces slice winners.
    {
        int s = loc;
        int lo = (int)(((long long)s * VDIM) / per);
        int hi = (int)(((long long)(s + 1) * VDIM) / per);
        float mv = -3.402823466e+38f;
        int mi = lo;
        for (int i = lo + tid; i < hi; i += NT) {
            float v = logb[i];
            if (v > mv || (v == mv && i < mi)) { mv = v; mi = i; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, mv, off);
            int oi = __shfl_down_sync(0xffffffffu, mi, off);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if ((tid & 31) == 0) { sval[tid >> 5] = mv; sidx[tid >> 5] = mi; }
        __syncthreads();
        if (tid < 32) {
            mv = (tid < NT / 32) ? sval[tid] : -3.402823466e+38f;
            mi = (tid < NT / 32) ? sidx[tid] : VDIM;
            for (int off = 4; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, mv, off);
                int oi = __shfl_down_sync(0xffffffffu, mi, off);
                if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
            }
            if (tid == 0) {
                amaxv[s * B + b] = mv;
                amaxi[s * B + b] = mi;
            }
        }
    }
    gbar(gcnt, ggen);
    if (loc == 0 && tid == 0) {
        float mv = -3.402823466e+38f;
        int mi = VDIM;
        for (int s = 0; s < per; ++s) {
            float v = amaxv[s * B + b];
            int i = amaxi[s * B + b];
            if (v > mv || (v == mv && i < mi)) { mv = v; mi = i; }
        }
        cur[b] = (long long)mi;
        tokm[(long long)step_i * B + b] = (long long)mi;
        pos[b] += 1;
        __threadfence_system();
    }
    gbar(gcnt, ggen);
    if (loc == 0 && tid == 0) flag[b] = (long long)(step_i + 1);
    }
}
"""


class Rtc:
    """Thin ctypes wrapper over libnvrtc + libcuda driver API."""

    def __init__(self):
        self.ok = False
        self.err = ""
        try:
            self._load()
            self.ok = True
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"

    def _load(self):
        cands = sorted(glob.glob(os.path.join(
            os.path.dirname(torch.__file__), "lib", "libnvrtc*")))
        cands += ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
        last = None
        for c in cands:
            try:
                self.nvrtc = ctypes.CDLL(c)
                break
            except OSError as e:
                last = e
        else:
            raise OSError(f"no libnvrtc: {last}")
        self.cuda = ctypes.CDLL("libcuda.so.1")
        for name in ("cuModuleLoadData", "cuModuleGetFunction",
                     "cuLaunchKernel", "cuMemHostAlloc",
                     "cuMemHostGetDevicePointer"):
            getattr(self.cuda, name)
        for name in ("nvrtcCreateProgram", "nvrtcCompileProgram",
                     "nvrtcGetPTXSize", "nvrtcGetPTX",
                     "nvrtcGetProgramLogSize", "nvrtcGetProgramLog",
                     "nvrtcDestroyProgram"):
            getattr(self.nvrtc, name)

    def compile(self, src, name, fn_name):
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), src.encode(), name.encode(), 0, None, None)
        if rc or not prog:
            raise RuntimeError(f"nvrtcCreateProgram rc={rc}")
        cap = torch.cuda.get_device_capability()
        arch = f"--gpu-architecture=compute_{cap[0]}{cap[1]}".encode()
        opts = (ctypes.c_char_p * 2)(arch, b"--std=c++11")
        rc = self.nvrtc.nvrtcCompileProgram(prog, 2, opts)
        if rc:
            sz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            log = ctypes.create_string_buffer(sz.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"compile {name}: rc={rc} {log.value[:400]!r}")
        sz = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz))
        ptx = ctypes.create_string_buffer(sz.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        mod = ctypes.c_void_p()
        rc = self.cuda.cuModuleLoadData(ctypes.byref(mod), ptx.raw)
        if rc or not mod:
            raise RuntimeError(f"cuModuleLoadData rc={rc}")
        fn = ctypes.c_void_p()
        rc = self.cuda.cuModuleGetFunction(
            ctypes.byref(fn), mod, fn_name.encode())
        if rc or not fn:
            raise RuntimeError(f"cuModuleGetFunction rc={rc}")
        return fn

    def launch(self, fn, grid, block, smem, args, coop=False):
        arr = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(a) for a in args])
        stream = torch.cuda.current_stream().cuda_stream
        if coop:
            rc = self.cuda.cuLaunchCooperativeKernel(
                fn, grid, 1, 1, block, 1, 1, smem,
                ctypes.c_void_p(stream), arr)
            if rc:
                raise RuntimeError(f"cuLaunchCooperativeKernel rc={rc}")
            return
        rc = self.cuda.cuLaunchKernel(
            fn, grid, 1, 1, block, 1, 1, smem,
            ctypes.c_void_p(stream), arr, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel rc={rc}")


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def i32(v):
    return ctypes.c_int(v)


def i64(v):
    return ctypes.c_longlong(v)


def f32(v):
    return ctypes.c_float(v)


class RtcKernels:
    """Compiled kernel set for the fused decode step."""

    def __init__(self):
        self.rtc = Rtc()
        self.rms = None
        if self.rtc.ok:
            self.rms = self.rtc.compile(RMS_SRC, "rms", "rms_k")
            self.rope = self.rtc.compile(ROPE_CACHE_SRC, "rope",
                                       "rope_cache_k")
            self.attn = self.rtc.compile(ATTN_SRC, "attn", "attn_k")
            self.silu = self.rtc.compile(SILU_SRC, "silu", "silu_k")
            self.mega = self.rtc.compile(ATTN_MEGA_SRC, "mega",
                                       "attn_mega_k")
            self.megafn = self.rtc.compile(MEGA_SRC, "stepall",
                                           "step_all_k")
            # cooperative variant: hardware grid.sync() barriers
            self.megacoop = None
            try:
                self.megacoop = self.rtc.compile(
                    "#define COOP 1\n" + MEGA_SRC, "stepallc",
                    "step_all_k")
            except Exception:
                pass
            self.nblk = torch.cuda.get_device_properties(
                0).multi_processor_count

    @property
    def ok(self):
        return self.rtc.ok and self.rms is not None

    def rms_norm(self, x, w, out, n, eps):
        self.rtc.launch(self.rms, x.shape[0], 256, 0,
                        [ptr(x), ptr(w), ptr(out), i32(n), f32(eps)])

    def rope_cache(self, qkv, qn, kn, cosb, sinb, qbuf, kc, vc, pos, cap, B):
        self.rtc.launch(self.rope, B * 48, 128, 0,
                        [ptr(qkv), ptr(qn), ptr(kn), ptr(cosb), ptr(sinb),
                         ptr(qbuf), ptr(kc), ptr(vc), ptr(pos),
                         i64(cap), f32(1e-6)])

    def attn(self, qbuf, kc, vc, pos, out, cap, B, maxs):
        smem = (maxs + 8) * 6 + 64
        self.rtc.launch(self.attn, B * 32, 128, smem,
                        [ptr(qbuf), ptr(kc), ptr(vc), ptr(pos), ptr(out),
                         i64(cap), f32(1.0 / 128 ** 0.5)])

    def attn_mega(self, qkv, qn, kn, cosb, sinb, kc, vc, pos, out,
                  cap, B, maxs):
        smem = (maxs + 8) * 6 + 64
        self.rtc.launch(self.mega, B * 8, 128, smem,
                        [ptr(qkv), ptr(qn), ptr(kn), ptr(cosb), ptr(sinb),
                         ptr(kc), ptr(vc), ptr(pos), ptr(out),
                         i64(cap), f32(1.0 / 128 ** 0.5), f32(1e-6)])

    def silu_mul(self, gu, out, I):
        n = gu.shape[0] * I
        self.rtc.launch(self.silu, (n + 127) // 128, 128, 0,
                        [ptr(gu), ptr(out), i32(I)])

    def host_map(self, nbytes):
        """Devicemapped host buffer -> (host ctypes array, device ptr)."""
        hp = ctypes.c_void_p()
        rc = self.rtc.cuda.cuMemHostAlloc(ctypes.byref(hp), nbytes, 0x03)
        if rc or not hp:
            raise RuntimeError(f"cuMemHostAlloc rc={rc}")
        dp = ctypes.c_void_p()
        rc = self.rtc.cuda.cuMemHostGetDevicePointer(
            ctypes.byref(dp), hp, 0)
        if rc or not dp:
            raise RuntimeError(f"cuMemHostGetDevicePointer rc={rc}")
        buf = (ctypes.c_longlong * (nbytes // 8)).from_address(hp.value)
        for i in range(nbytes // 8):
            buf[i] = 0
        return buf, dp.value

    def step_all(self, lw, embed, finw, cost, sint, pos, cur, hid, hbuf,
                 qkv, obuf, gu, logits, amaxv, amaxi, cnt, gen,
                 flag_dev, tokm_dev, B, NL, ntok, eps, cap, bench=0):
        smem = (cap + 8) * 6 + 512
        if self.megacoop is not None:
            try:
                self.rtc.launch(self.megacoop, self.nblk, 512, smem,
                                [ptr(lw), ptr(embed), ptr(finw), ptr(cost),
                                 ptr(sint), ptr(pos), ptr(cur), ptr(hid),
                                 ptr(hbuf), ptr(qkv), ptr(obuf), ptr(gu),
                                 ptr(logits), ptr(amaxv), ptr(amaxi),
                                 ptr(cnt), ptr(gen),
                                 ctypes.c_void_p(flag_dev),
                                 ctypes.c_void_p(tokm_dev),
                                 i32(B), i32(NL), i32(ntok), f32(eps),
                                 i64(cap), i32(bench)], coop=True)
                self.coop_ok = True
                return
            except Exception:
                self.coop_ok = False
                self.megacoop = None
        self.rtc.launch(self.megafn, self.nblk, 512, smem,
                        [ptr(lw), ptr(embed), ptr(finw), ptr(cost),
                         ptr(sint), ptr(pos), ptr(cur), ptr(hid),
                         ptr(hbuf), ptr(qkv), ptr(obuf), ptr(gu),
                         ptr(logits), ptr(amaxv), ptr(amaxi),
                         ptr(cnt), ptr(gen),
                         ctypes.c_void_p(flag_dev),
                         ctypes.c_void_p(tokm_dev),
                         i32(B), i32(NL), i32(ntok), f32(eps), i64(cap),
                         i32(bench)])
