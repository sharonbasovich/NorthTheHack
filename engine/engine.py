"""Optimized Qwen3 4B engine: native-exact prefill into a static KV cache,
then a single CUDA-graph-captured decode step built from fused Triton
kernels + cuBLAS GEMMs.

Same math as the baseline, reorganized for speed:
- prefill runs the pinned Transformers layers once, writing K/V directly into
  preallocated contiguous cache buffers (no per-step concatenation);
- decode replays a captured graph per step: embedding, fused QKV GEMM,
  one Triton kernel for per-head RMSNorm + RoPE + K/V cache scatter, a
  flash-decode Triton attention (K/V read once per KV head, fp32 online
  softmax), fused gate/up GEMM, fused SiLU-mul, final norm, tied LM head,
  argmax written back into the input buffer;
- RoPE row, cache slot, and attention length all come from a device-side
  position counter, so the graph stays correct as it advances;
- per-step host readback is one tiny pinned-memory copy outside the graph.

Numerics match native: fp32 norm accumulation with the same cast placement,
bf16 rope tables identical to the model's own rotary module, fp32 softmax,
lowest-index argmax.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from kernels import fused as FK
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

H = 2560
NQ = 32
NKV = 8
GROUP = NQ // NKV
D = 128
I = 9728
NL = 36
EPS = 1e-6
SCALE = 1.0 / (D ** 0.5)
NEG_INF = float("-inf")
V = 151936


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Qwen3RMSNorm semantics: fp32 reduce, cast to bf16, then weight mul."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w * (xf * torch.rsqrt(var + EPS)).to(x.dtype)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _PrefillCache:
    """Minimal cache API for Transformers 4.51.3 attention during prefill.

    Writes into preallocated per-layer buffers and returns the valid prefix.
    ``seen`` advances once per forward (layer_idx == 0), matching DynamicCache.
    """

    def __init__(self, kc, vc):
        self.kc = kc
        self.vc = vc
        self.seen = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        t = key_states.shape[2]
        if layer_idx == 0:
            self.seen += t
        p = self.seen - t
        self.kc[layer_idx][:, :, p : p + t].copy_(key_states)
        self.vc[layer_idx][:, :, p : p + t].copy_(value_states)
        return self.kc[layer_idx][:, :, : p + t], self.vc[layer_idx][:, :, : p + t]

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def get_usable_length(self, new_length, layer_idx=0):
        return self.seen

    def get_max_cache_shape(self):
        return -1

    def reorder_cache(self, beam_idx):
        raise NotImplementedError


class _State:
    """Per-(batch, capacity) buffers plus the captured decode graph."""

    def __init__(self, engine, batch, capacity):
        dev = engine.dev
        self.B = batch
        self.S = capacity
        self.kc = [
            torch.zeros(batch, NKV, capacity, D, dtype=torch.bfloat16, device=dev)
            for _ in range(NL)
        ]
        self.vc = [
            torch.zeros(batch, NKV, capacity, D, dtype=torch.bfloat16, device=dev)
            for _ in range(NL)
        ]
        self.cur = torch.zeros(batch, 1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        self.mask = torch.full(
            (capacity,), NEG_INF, dtype=torch.bfloat16, device=dev
        )
        positions = torch.arange(capacity, device=dev, dtype=torch.float32)
        inv = engine.model.model.rotary_emb.inv_freq.float().to(dev)
        freqs = torch.outer(positions, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(torch.bfloat16)
        self.sin = emb.sin().to(torch.bfloat16)
        self.pin = torch.empty(batch, dtype=torch.int64, pin_memory=True)
        # fused-path scratch buffers
        self.x = torch.zeros(batch, H, dtype=torch.bfloat16, device=dev)
        self.h = torch.zeros(batch, H, dtype=torch.bfloat16, device=dev)
        self.h2 = torch.zeros(batch, H, dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros(batch, (NQ + 2 * NKV) * D, dtype=torch.bfloat16, device=dev)
        self.qe = torch.zeros(batch, NQ * D, dtype=torch.bfloat16, device=dev)
        self.o4 = torch.zeros(batch, NQ * D, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(batch, H, dtype=torch.bfloat16, device=dev)
        self.gu = torch.zeros(batch, 2 * I, dtype=torch.bfloat16, device=dev)
        self.m = torch.zeros(batch, I, dtype=torch.bfloat16, device=dev)
        self.dn = torch.zeros(batch, H, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(batch, V, dtype=torch.bfloat16, device=dev)
        self.graph = None


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dev = "cuda:0"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(self.dev)
        )
        base = self.model.model
        self.embed_w = base.embed_tokens.weight
        self.lm_w = self.model.lm_head.weight
        self.fin_w = base.norm.weight
        self._step_slow_only = False
        self.layers = []
        for layer in base.layers:
            a = layer.self_attn
            self.layers.append(
                {
                    "wqkv": torch.cat(
                        [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], dim=0
                    ).contiguous(),
                    "wo": a.o_proj.weight,
                    "wgu": torch.cat(
                        [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0
                    ).contiguous(),
                    "wd": layer.mlp.down_proj.weight,
                    "ln_in": layer.input_layernorm.weight,
                    "ln_post": layer.post_attention_layernorm.weight,
                    "qn": a.q_norm.weight,
                    "kn": a.k_norm.weight,
                }
            )
        self.states = {}

    # ------------------------------------------------------------------
    # Decode step, fused-Triton version.
    # ------------------------------------------------------------------
    def _decode_step_fast(self, st: _State) -> None:
        torch.index_select(self.embed_w, 0, st.cur[:, 0], out=st.x)
        FK.rmsnorm(st.x, self.layers[0]["ln_in"], st.h, EPS)
        for i, w in enumerate(self.layers):
            torch.matmul(st.h, w["wqkv"].t(), out=st.qkv)
            FK.qknorm_rope_cache(
                st.qkv, w["qn"], w["kn"], st.cos, st.sin,
                st.kc[i], st.vc[i], st.qe, st.pos, EPS, NQ, NKV, D,
            )
            FK.attn_decode(
                st.qe, st.kc[i], st.vc[i], st.o4, st.pos,
                NKV, GROUP, D, SCALE,
            )
            torch.matmul(st.o4, w["wo"].t(), out=st.att)
            FK.add_rmsnorm(st.x, st.att, st.h2, w["ln_post"], EPS)
            torch.matmul(st.h2, w["wgu"].t(), out=st.gu)
            FK.silu_mul(st.gu, st.m, I)
            torch.matmul(st.m, w["wd"].t(), out=st.dn)
            next_w = (
                self.layers[i + 1]["ln_in"] if i + 1 < NL else self.fin_w
            )
            FK.add_rmsnorm(st.x, st.dn, st.h, next_w, EPS)
        torch.matmul(st.h, self.lm_w.t(), out=st.logits)
        tok = st.logits.argmax(dim=-1)
        st.cur.copy_(tok.view(st.B, 1))
        st.pos.add_(1)

    # ------------------------------------------------------------------
    # Decode step, plain-torch fallback (same math, more kernels).
    # ------------------------------------------------------------------
    def _decode_step_slow(self, st: _State) -> None:
        B = st.B
        x = F.embedding(st.cur, self.embed_w)  # [B,1,H]
        cos = st.cos.index_select(0, st.pos).view(1, 1, 1, D)
        sin = st.sin.index_select(0, st.pos).view(1, 1, 1, D)
        pos1 = st.pos.view(1)
        for i, w in enumerate(self.layers):
            h = _rms(x, w["ln_in"]).view(B, H)
            qkv = h @ w["wqkv"].t()
            q = qkv[:, : NQ * D].view(B, 1, NQ, D)
            k = qkv[:, NQ * D : NQ * D + NKV * D].view(B, 1, NKV, D)
            v = qkv[:, NQ * D + NKV * D :].view(B, 1, NKV, D)
            qn = _rms(q, w["qn"])
            kn = _rms(k, w["kn"])
            qe = qn * cos + _rot_half(qn) * sin
            ke = kn * cos + _rot_half(kn) * sin
            st.kc[i].index_copy_(2, pos1, ke.transpose(1, 2))
            st.vc[i].index_copy_(2, pos1, v.transpose(1, 2))
            st.mask.index_fill_(0, pos1, 0.0)
            qg = qe.view(B, NKV, GROUP, D)
            scores = torch.matmul(qg, st.kc[i].transpose(-1, -2)) * SCALE
            scores = scores + st.mask.view(1, 1, 1, st.S)
            p = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
            o = torch.matmul(p, st.vc[i]).view(B, 1, NQ * D)
            x = x + (o.view(B, NQ * D) @ w["wo"].t()).view(B, 1, H)
            h2 = _rms(x, w["ln_post"]).view(B, H)
            gu = h2 @ w["wgu"].t()
            m = F.silu(gu[:, :I]) * gu[:, I:]
            x = x + (m @ w["wd"].t()).view(B, 1, H)
        x = _rms(x, self.fin_w)
        logits = x.view(B, H) @ self.lm_w.t()
        tok = logits.argmax(dim=-1)
        st.cur.copy_(tok.view(B, 1))
        st.pos.add_(1)

    @torch.inference_mode()
    def _prefill(self, st: _State, ids: torch.Tensor) -> torch.Tensor:
        """Native-layer prefill writing into the static cache. Returns [B]."""
        base = self.model.model
        length = ids.shape[1]
        position_ids = torch.arange(length, device=self.dev).unsqueeze(0)
        cache_position = torch.arange(length, device=self.dev)
        x = base.embed_tokens(ids)
        pos_emb = base.rotary_emb(x, position_ids)
        cache = _PrefillCache(st.kc, st.vc)
        for layer in base.layers:
            x = layer(
                x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=pos_emb,
            )[0]
        x = base.norm(x)
        logits = self.model.lm_head(x[:, -1, :])
        return logits.argmax(dim=-1)

    def _get_state(self, batch: int, capacity: int) -> _State:
        key = (batch, capacity)
        st = self.states.get(key)
        if st is None:
            st = _State(self, batch, capacity)
            self.states[key] = st
        return st

    def _reset_decode(self, st: _State, prompt_len: int, first: torch.Tensor) -> None:
        st.mask.fill_(NEG_INF)
        st.mask[:prompt_len] = 0.0
        st.pos.fill_(prompt_len)
        st.cur.copy_(first.view(st.B, 1))

    def _pick_step(self):
        if _HAS_TRITON and not getattr(self, "_step_slow_only", False):
            return self._decode_step_fast
        return self._decode_step_slow

    def _self_check(self, st: _State) -> None:
        """Cross-validate the fused step against the torch step on live state.

        Runs the fast step once, rewinds pos/cur, runs the slow step, and
        compares the emitted token. Any exception or disagreement disables the
        fused path for the rest of the process. Cheap: two extra decode steps,
        only once per (batch, capacity) state.
        """
        if self._step_slow_only or not _HAS_TRITON:
            return
        try:
            c0 = st.cur.clone()
            p0 = st.pos.clone()
            self._decode_step_fast(st)
            tok_fast = st.cur.clone()
            st.cur.copy_(c0)
            st.pos.copy_(p0)
            self._decode_step_slow(st)
            tok_slow = st.cur.clone()
            st.cur.copy_(c0)
            st.pos.copy_(p0)
            if not torch.equal(tok_fast, tok_slow):
                self._step_slow_only = True
        except Exception:
            self._step_slow_only = True

    def _capture(self, st: _State, step, iters: int) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(iters):
                step(st)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step(st)
        st.graph = g

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.dev)
        B, L = ids.shape
        st = self._get_state(B, L + max_new_tokens)
        first = self._prefill(st, ids)
        self._reset_decode(st, L, first)

        if st.graph is None:
            self._self_check(st)
            self._reset_decode(st, L, first)
            # warmup+capture need `iters+1` spare cache slots past pos=L.
            spare = max_new_tokens - 2
            iters = max(0, min(2, spare - 1))
            try:
                if spare >= 1:
                    self._capture(st, self._pick_step(), iters)
                else:
                    st.graph = False
            except Exception:
                if not self._step_slow_only:
                    self._step_slow_only = True
                    self._reset_decode(st, L, first)
                    try:
                        if spare >= 1:
                            self._capture(st, self._decode_step_slow, iters)
                        else:
                            st.graph = False
                    except Exception:
                        st.graph = False
                else:
                    st.graph = False
            self._reset_decode(st, L, first)

        use_fast = not self._step_slow_only and _HAS_TRITON
        yield first.tolist()
        for _ in range(max_new_tokens - 1):
            if st.graph:
                st.graph.replay()
            else:
                (self._decode_step_fast if use_fast else self._decode_step_slow)(st)
            st.pin.copy_(st.cur[:, 0], non_blocking=True)
            torch.cuda.synchronize()
            yield st.pin.tolist()
