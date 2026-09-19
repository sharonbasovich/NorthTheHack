"""Runtime-compiled fused decode kernels via NVRTC + CUDA driver API.

The eval box has no Triton (import fails), no nvcc toolchain, and gVisor
makes CUDA graphs worthless (nvproxy replays captured nodes through the
same trapped-ioctl path, so graph launch costs as much as eager). What it
DOES have: libnvrtc.so bundled inside the torch wheel and libcuda via the
driver. ctypes -> nvrtcCompileProgram -> cuModuleLoadData -> cuLaunchKernel
gives us real fused kernels with ~5us launches instead of ~30 eager ops
per layer.

Numerics contract (must match the reference decode step bit-for-bit in
cast placement):
  - RMSNorm: fp32 accumulate of x^2 mean, rsqrt in fp32, product cast to
    bf16, then multiplied by the (bf16) weight in bf16.
  - scores: fp32 dot of bf16 q,k (matches cublas bf16->fp32 accumulation),
    times scale, -inf above pos, fp32 softmax, cast p to bf16.
  - attention out: fp32 accumulate of bf16(p) * bf16(v), cast bf16.
"""
import ctypes
import glob
import os
import torch

RMS_SRC = r"""
#include <cuda_bf16.h>
extern "C" __global__ void rms_k(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ w,
                                 __nv_bfloat16* __restrict__ out,
                                 int n, float eps) {
    // one block per row
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)row * n;
    __nv_bfloat16* orow = out + (long long)row * n;
    __shared__ float ssum[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        acc += v * v;
    }
    // warp + block reduce
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
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __bfloat162float(xr[i]) * r;
        __nv_bfloat16 t = __float2bfloat16(v);
        orow[i] = __hmul(w[i], t);
    }
}
"""

ROPE_CACHE_SRC = r"""
#include <cuda_bf16.h>
// grid.x = B * 48  (heads: 0..31 q, 32..39 k, 40..47 v), block = 128 threads
extern "C" __global__ void rope_cache_k(
        const __nv_bfloat16* __restrict__ qkv,   // [B, 48*128]
        const __nv_bfloat16* __restrict__ qn,    // [128]
        const __nv_bfloat16* __restrict__ kn,    // [128]
        const __nv_bfloat16* __restrict__ cosb,  // [B, 128] bf16
        const __nv_bfloat16* __restrict__ sinb,  // [B, 128] bf16
        __nv_bfloat16* __restrict__ qbuf,        // [B, 32*128]
        __nv_bfloat16* __restrict__ kc,          // flat [B*8*cap,128]
        __nv_bfloat16* __restrict__ vc,          // flat [B*8*cap,128]
        const long* __restrict__ pos,            // [B]
        long cap, float eps) {
    int b = blockIdx.x / 48;
    int h = blockIdx.x % 48;
    int d = threadIdx.x;
    const __nv_bfloat16* src = qkv + (long long)b * 48 * 128 + h * 128;
    if (h >= 40) {
        // v head: straight copy into the cache at (b, h-40, pos)
        long j = h - 40;
        long idx = ((long)b * 8 + j) * cap + pos[b];
        vc[idx * 128 + d] = src[d];
        return;
    }
    const __nv_bfloat16* w = (h < 32) ? qn : kn;
    __shared__ float ssum[4];
    float xv = __bfloat162float(src[d]);
    float acc = xv * xv;
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = ssum[0] + ssum[1] + ssum[2] + ssum[3];
    __shared__ __nv_bfloat16 srope[128];
    float r = rsqrtf(tot / 128.f + eps);
    // norm then cast to bf16 then *w in bf16 (reference cast order)
    __nv_bfloat16 t = __float2bfloat16(xv * r);
    srope[d] = __hmul(w[d], t);
    __syncthreads();
    __nv_bfloat16 me = srope[d];
    __nv_bfloat16 rot = (d < 64) ? __hneg(srope[d + 64]) : srope[d - 64];
    __nv_bfloat16 ob = __hadd(__hmul(me, cosb[b * 128 + d]),
                              __hmul(rot, sinb[b * 128 + d]));
    if (h < 32) {
        qbuf[((long long)b * 32 + h) * 128 + d] = ob;
    } else {
        long j = h - 32;
        long idx = ((long)b * 8 + j) * cap + pos[b];
        kc[idx * 128 + d] = ob;
    }
}
"""

ATTN_SRC = r"""
#include <cuda_bf16.h>
// grid.x = B * 32 (one block per q head). block = 128 threads.
// phase1: each thread strides over cache rows computing q.k*scale (fp32,
// bf16 inputs) into shared scores; -inf for s > pos.
// phase2: block max, fp32 exp/sum, p cast to bf16 in shared.
// phase3: thread d accumulates sum_s p_bf16 * v_bf16 in fp32 -> bf16 out.
extern "C" __global__ void attn_k(
        const __nv_bfloat16* __restrict__ qbuf,  // [B, 32*128]
        const __nv_bfloat16* __restrict__ kc,    // flat [B*8*cap,128]
        const __nv_bfloat16* __restrict__ vc,
        const long* __restrict__ pos,            // [B]
        __nv_bfloat16* __restrict__ out,         // [B, 32*128]
        long cap, float scale) {
    int b = blockIdx.x / 32;
    int h = blockIdx.x % 32;
    int j = h >> 2;
    long p = pos[b];
    const __nv_bfloat16* q = qbuf + ((long long)b * 32 + h) * 128;
    const __nv_bfloat16* kbase = kc + ((long)b * 8 + j) * cap * 128;
    const __nv_bfloat16* vbase = vc + ((long)b * 8 + j) * cap * 128;
    __shared__ float qs[128];
    __shared__ float red[32];
    extern __shared__ float scores[];
    if (threadIdx.x < 128) qs[threadIdx.x] = __bfloat162float(q[threadIdx.x]);
    __syncthreads();
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x) {
        const __nv_bfloat16* krow = kbase + (long)s * 128;
        float dot = 0.f;
        for (int d = 0; d < 128; ++d)
            dot += qs[d] * __bfloat162float(krow[d]);
        scores[s] = dot * scale;
    }
    __syncthreads();
    // block max
    float mx = -INFINITY;
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
    // store bf16 p into the tail of the dynamic smem buffer
    __nv_bfloat16* pbv = (__nv_bfloat16*)(scores + (p + 2));
    for (int s = threadIdx.x; s <= (int)p; s += blockDim.x)
        pbv[s] = __float2bfloat16(scores[s] / sm);
    __syncthreads();
    int d = threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s <= (int)p; ++s)
        acc += __bfloat162float(pbv[s]) * __bfloat162float(vbase[(long)s * 128 + d]);
    out[((long long)b * 32 + h) * 128 + d] = __float2bfloat16(acc);
}
"""

SILU_SRC = r"""
#include <cuda_bf16.h>
// gu: [B, 2*I] row-major; out[b,i] = silu(gu[b,i]) * gu[b,I+i]
extern "C" __global__ void silu_k(const __nv_bfloat16* __restrict__ gu,
                                  __nv_bfloat16* __restrict__ out, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / I, i = idx % I;
    float a = __bfloat162float(gu[(long)b * 2 * I + i]);
    float g = __bfloat162float(gu[(long)b * 2 * I + I + i]);
    float s = a / (1.f + expf(-a));
    out[idx] = __hmul(__float2bfloat16(s), __float2bfloat16(g));
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
                     "cuLaunchKernel"):
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

    def launch(self, fn, grid, block, smem, args):
        arr = (ctypes.c_void_p * len(args))(
            *[ctypes.addressof(a) for a in args])
        stream = torch.cuda.current_stream().cuda_stream
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
            self.rope = self.rtc.compile(ROPE_CACHE_SRC, "rope", "rope_cache_k")
            self.attn = self.rtc.compile(ATTN_SRC, "attn", "attn_k")
            self.silu = self.rtc.compile(SILU_SRC, "silu", "silu_k")

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
        # smem: fp32 scores[maxs] + bf16 p[maxs] + slack
        smem = (maxs + 8) * 6 + 64
        self.rtc.launch(self.attn, B * 32, 128, smem,
                        [ptr(qbuf), ptr(kc), ptr(vc), ptr(pos), ptr(out),
                         i64(cap), f32(1.0 / 128 ** 0.5)])

    def silu_mul(self, gu, out, I):
        n = gu.shape[0] * I
        self.rtc.launch(self.silu, (n + 127) // 128, 128, 0,
                        [ptr(gu), ptr(out), i32(I)])
