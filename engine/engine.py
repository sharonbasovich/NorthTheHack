"""Optimized Qwen3 4B engine: native-exact prefill into a static KV cache,
then a single CUDA-graph-captured decode step.

Same math as the baseline, reorganized for speed:
- prefill runs the pinned Transformers layers once, writing K/V directly into
  preallocated contiguous cache buffers (no per-step concatenation);
- decode replays a captured graph per step: embedding, fused QKV GEMM,
  per-head RMSNorm, RoPE from precomputed bf16 tables, grouped-query
  attention over the padded cache with an additive mask, fused gate/up GEMM,
  final norm, tied LM head, argmax written back into the input buffer;
- per-step host readback is one tiny pinned-memory copy outside the graph.

Numerics match native: fp32 norm accumulation with the same cast placement,
bf16 rope tables identical to the model's own rotary module, fp32 softmax.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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
    """Per-(batch, prompt, total length) buffers and the captured graph."""

    def __init__(self, engine, batch, capacity):
        dev = engine.dev
        self.B = batch
        self.S = capacity
        # Zero-filled: masked-out scores must be 0 + -inf, never NaN from
        # garbage bytes in uninitialized memory.
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

    def _decode_step(self, st: _State) -> None:
        """One decode step over static buffers; safe inside a CUDA graph.

        On entry ``st.cur`` holds the last emitted token at absolute position
        ``st.pos``; on exit it holds the new greedy token and ``st.pos`` is
        incremented. Reads/writes are ordered on one stream.
        """
        B = st.B
        x = F.embedding(st.cur, self.embed_w)  # [B,1,H]
        cos = st.cos.index_select(0, st.pos).view(1, 1, 1, D)
        sin = st.sin.index_select(0, st.pos).view(1, 1, 1, D)
        pos1 = st.pos.view(1)
        for i, w in enumerate(self.layers):
            h = _rms(x, w["ln_in"]).view(B, H)
            qkv = h @ w["wqkv"].t()  # [B, NQ*D + 2*NKV*D]
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
            gu = h2 @ w["wgu"].t()  # [B, 2*I]
            m = F.silu(gu[:, :I]) * gu[:, I:]
            x = x + (m @ w["wd"].t()).view(B, 1, H)
        x = _rms(x, self.fin_w)
        logits = x.view(B, H) @ self.lm_w.t()  # [B, V]
        tok = logits.argmax(dim=-1)
        st.cur.copy_(tok.view(B, 1))
        st.pos.add_(1)

    @torch.inference_mode()
    def _prefill(self, st: _State, ids: torch.Tensor) -> torch.Tensor:
        """Native-layer prefill writing into the static cache. Returns [B] token."""
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

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.dev)
        B, L = ids.shape
        st = self._get_state(B, L + max_new_tokens)
        first = self._prefill(st, ids)
        self._reset_decode(st, L, first)

        if st.graph is None:
            try:
                # Stabilize allocator/cuBLAS heuristics on a side stream, then
                # capture one decode step; capture does not execute kernels.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._decode_step(st)
                torch.cuda.current_stream().wait_stream(s)
                self._reset_decode(st, L, first)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._decode_step(st)
                st.graph = g
                self._reset_decode(st, L, first)
            except Exception:
                st.graph = False  # eager decode still beats the baseline
                self._reset_decode(st, L, first)

        yield first.tolist()
        for _ in range(max_new_tokens - 1):
            if st.graph:
                st.graph.replay()
            else:
                self._decode_step(st)
            st.pin.copy_(st.cur[:, 0], non_blocking=True)
            torch.cuda.synchronize()
            yield st.pin.tolist()
