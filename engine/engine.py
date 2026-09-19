"""Optimized Qwen3 4B engine: native-exact prefill into a static KV cache,
then speculative decode with exact verification, captured as CUDA graphs.

Pipeline:
- prefill runs the pinned Transformers layers once, writing K/V directly into
  preallocated contiguous cache buffers;
- decode is a captured graph of fused Triton kernels + cuBLAS GEMMs;
- a verify pass runs R = K+1 rows per sequence (the confirmed last token plus
  K draft tokens from an n-gram prompt-lookup), producing per-row argmaxes;
  the leading run where each draft equals the previous row's argmax is
  accepted — every emitted token is the model's greedy choice on its true
  prefix, so outputs are identical to sequential greedy decode;
- positions live in a device-side per-batch counter, so a captured graph
  replays correctly as positions advance by variable amounts;
- per-step host traffic is two tiny pinned copies (draft inputs in,
  emitted tokens out).

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

K_DRAFT = 4          # draft tokens per verify pass
R = K_DRAFT + 1      # rows per sequence in the verify pass
NGRAM_SIZES = (4, 3, 2)


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Qwen3RMSNorm semantics: fp32 reduce, cast to bf16, then weight mul."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w * (xf * torch.rsqrt(var + EPS)).to(x.dtype)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _Ngram:
    """Per-row n-gram index over prompt + emitted tokens for prompt-lookup
    drafting. Maps each n-gram to the positions it starts at, most recent last."""

    def __init__(self, hist):
        self.hist = list(hist)
        self.index = {}
        for i in range(len(self.hist)):
            self._add(i)

    def _add(self, i):
        h = self.hist
        for n in NGRAM_SIZES:
            if i + n <= len(h):
                self.index.setdefault(tuple(h[i : i + n]), []).append(i)

    def append(self, tok):
        self.hist.append(tok)
        self._add(len(self.hist) - 1)

    def draft(self, k):
        """Up to k continuation tokens after the most recent earlier occurrence
        of the longest matching suffix; [] when nothing matches."""
        L = len(self.hist)
        for n in NGRAM_SIZES:
            if L <= n:
                continue
            cands = self.index.get(tuple(self.hist[L - n :]))
            if not cands:
                continue
            for i in reversed(cands):
                if i + n < L:  # occurrence with a following token
                    out = self.hist[i + n : i + n + k]
                    if out:
                        return out
        return []


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
    """Per-(batch, prompt+output length) buffers plus captured graphs.

    `rows` scratch is sized for R rows per sequence (the verify pass);
    the single-token step uses the leading B rows.
    """

    def __init__(self, engine, batch, capacity):
        dev = engine.dev
        self.B = batch
        self.S = capacity           # includes slack for overshoot writes
        self.kc = [
            torch.zeros(batch, NKV, capacity, D, dtype=torch.bfloat16, device=dev)
            for _ in range(NL)
        ]
        self.vc = [
            torch.zeros(batch, NKV, capacity, D, dtype=torch.bfloat16, device=dev)
            for _ in range(NL)
        ]
        self.cur = torch.zeros(batch, 1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.srange = torch.arange(capacity, device=dev)
        self.ii = torch.arange(batch, device=dev).view(batch, 1).expand(batch, NKV)
        self.jj = torch.arange(NKV, device=dev).view(1, NKV).expand(batch, NKV)
        positions = torch.arange(capacity, device=dev, dtype=torch.float32)
        inv = engine.model.model.rotary_emb.inv_freq.float().to(dev)
        freqs = torch.outer(positions, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(torch.bfloat16)
        self.sin = emb.sin().to(torch.bfloat16)
        self.pin = torch.empty(batch, dtype=torch.int64, pin_memory=True)
        # fused-path scratch, sized for R rows/sequence
        n = batch * R
        self.x = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        self.h = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        self.h2 = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        self.qkv = torch.zeros(n, (NQ + 2 * NKV) * D, dtype=torch.bfloat16, device=dev)
        self.qe = torch.zeros(n, NQ * D, dtype=torch.bfloat16, device=dev)
        self.o4 = torch.zeros(n, NQ * D, dtype=torch.bfloat16, device=dev)
        self.att = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        self.gu = torch.zeros(n, 2 * I, dtype=torch.bfloat16, device=dev)
        self.mlp = torch.zeros(n, I, dtype=torch.bfloat16, device=dev)
        self.dn = torch.zeros(n, H, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(n, V, dtype=torch.bfloat16, device=dev)
        # speculative verify inputs / outputs
        self.inp = torch.zeros(batch, R, dtype=torch.int64, device=dev)
        self.inp_pin = torch.empty(batch, R, dtype=torch.int64, pin_memory=True)
        self.emit_dev = torch.zeros(batch, R + 1, dtype=torch.int64, device=dev)
        self.emit_pin = torch.empty(batch, R + 1, dtype=torch.int64, pin_memory=True)
        self.graph = None            # verify-pass graph (R rows)
        self.graph1 = None           # plain decode graph (R=1 rows)
        self.spec_cooldown = 0       # passes to run decode-only (weak drafting)
        self.spec_window = []        # recent emit counts for adaptivity


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
    # Verify pass: R rows per sequence. Emits 1..R tokens/row into emit_dev.
    # ------------------------------------------------------------------
    def _decode_step_spec(self, st: _State) -> None:
        B = st.B
        torch.index_select(self.embed_w, 0, st.inp.view(-1), out=st.x)
        FK.rmsnorm(st.x, self.layers[0]["ln_in"], st.h, EPS)
        for i, w in enumerate(self.layers):
            torch.matmul(st.h, w["wqkv"].t(), out=st.qkv)
            FK.qknorm_rope_cache(
                st.qkv, w["qn"], w["kn"], st.cos, st.sin,
                st.kc[i], st.vc[i], st.qe, st.pos, EPS, NQ, NKV, D, R,
            )
            FK.attn_decode(
                st.qe, st.kc[i], st.vc[i], st.o4, st.pos,
                NKV, GROUP, D, SCALE, R,
            )
            torch.matmul(st.o4, w["wo"].t(), out=st.att)
            FK.add_rmsnorm(st.x, st.att, st.h2, w["ln_post"], EPS)
            torch.matmul(st.h2, w["wgu"].t(), out=st.gu)
            FK.silu_mul(st.gu, st.mlp, I)
            torch.matmul(st.mlp, w["wd"].t(), out=st.dn)
            next_w = (
                self.layers[i + 1]["ln_in"] if i + 1 < NL else self.fin_w
            )
            FK.add_rmsnorm(st.x, st.dn, st.h, next_w, EPS)
        torch.matmul(st.h, self.lm_w.t(), out=st.logits)
        am = st.logits.view(B, R, V).argmax(dim=-1)       # [B,R] token ids
        matched = (am[:, :-1] == st.inp[:, 1:]).to(torch.int64)
        m = matched.cumprod(dim=1).sum(dim=1) + 1        # [B], 1..R
        st.pos.add_(m)
        torch.cat([am, m.view(B, 1)], dim=1, out=st.emit_dev)

    # ------------------------------------------------------------------
    # Single-token fused step (same kernels, R=1 view of the buffers).
    # ------------------------------------------------------------------
    def _decode_step_fast(self, st: _State) -> None:
        B = st.B
        x = st.x[:B]
        h = st.h[:B]
        torch.index_select(self.embed_w, 0, st.cur[:, 0], out=x)
        FK.rmsnorm(x, self.layers[0]["ln_in"], h, EPS)
        for i, w in enumerate(self.layers):
            torch.matmul(h, w["wqkv"].t(), out=st.qkv[:B])
            FK.qknorm_rope_cache(
                st.qkv[:B], w["qn"], w["kn"], st.cos, st.sin,
                st.kc[i], st.vc[i], st.qe[:B], st.pos, EPS, NQ, NKV, D, 1,
            )
            FK.attn_decode(
                st.qe[:B], st.kc[i], st.vc[i], st.o4[:B], st.pos,
                NKV, GROUP, D, SCALE, 1,
            )
            torch.matmul(st.o4[:B], w["wo"].t(), out=st.att[:B])
            FK.add_rmsnorm(x, st.att[:B], st.h2[:B], w["ln_post"], EPS)
            torch.matmul(st.h2[:B], w["wgu"].t(), out=st.gu[:B])
            FK.silu_mul(st.gu[:B], st.mlp[:B], I)
            torch.matmul(st.mlp[:B], w["wd"].t(), out=st.dn[:B])
            next_w = (
                self.layers[i + 1]["ln_in"] if i + 1 < NL else self.fin_w
            )
            FK.add_rmsnorm(x, st.dn[:B], h, next_w, EPS)
        torch.matmul(h, self.lm_w.t(), out=st.logits[:B])
        tok = st.logits[:B].argmax(dim=-1)
        st.cur.copy_(tok.view(B, 1))
        st.pos.add_(1)

    # ------------------------------------------------------------------
    # Plain-torch single-token fallback (same math, more kernels).
    # ------------------------------------------------------------------
    def _decode_step_slow(self, st: _State) -> None:
        B = st.B
        x = F.embedding(st.cur, self.embed_w)  # [B,1,H]
        cos = st.cos.index_select(0, st.pos).view(B, 1, 1, D)
        sin = st.sin.index_select(0, st.pos).view(B, 1, 1, D)
        valid = st.srange[None, :] <= st.pos[:, None]
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
            st.kc[i].index_put_(
                (st.ii, st.jj, st.pos.view(B, 1).expand(B, NKV)), ke[:, 0]
            )
            st.vc[i].index_put_(
                (st.ii, st.jj, st.pos.view(B, 1).expand(B, NKV)), v[:, 0]
            )
            qg = qe.view(B, NKV, GROUP, D)
            scores = torch.matmul(qg, st.kc[i].transpose(-1, -2)) * SCALE
            scores.masked_fill_(~valid[:, None, None, :], NEG_INF)
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

    def _get_state(self, batch: int, slack: int) -> _State:
        key = (batch, slack)
        st = self.states.get(key)
        if st is None:
            st = _State(self, batch, slack)
            self.states[key] = st
        return st

    def _reset_decode(self, st: _State, prompt_len: int, first: torch.Tensor) -> None:
        st.pos.fill_(prompt_len)
        st.cur.copy_(first.view(st.B, 1))

    def _pick_step(self):
        if _HAS_TRITON and not self._step_slow_only:
            return self._decode_step_fast
        return self._decode_step_slow

    def _self_check(self, st: _State) -> None:
        """Cross-validate the verify and fused steps against the torch step.

        Runs each on the same (cur, pos), rewinds, and compares the first
        emitted token. Any exception or disagreement disables fused paths.
        Cheap: a handful of extra steps, once per state.
        """
        if self._step_slow_only or not _HAS_TRITON:
            return
        try:
            c0 = st.cur.clone()
            p0 = st.pos.clone()

            # verify pass, all-junk drafts: only the first emitted token counts
            st.inp.fill_(0)
            st.inp[:, 0] = c0[:, 0]
            self._decode_step_spec(st)
            tok_spec = st.emit_dev[:, 0].clone()

            st.cur.copy_(c0)
            st.pos.copy_(p0)
            self._decode_step_fast(st)
            tok_fast = st.cur.clone()

            st.cur.copy_(c0)
            st.pos.copy_(p0)
            self._decode_step_slow(st)
            tok_slow = st.cur.clone()

            st.cur.copy_(c0)
            st.pos.copy_(p0)
            if not (torch.equal(tok_spec.view(-1), tok_slow.view(-1))
                    and torch.equal(tok_fast.view(-1), tok_slow.view(-1))):
                self._step_slow_only = True
        except Exception:
            self._step_slow_only = True

    def _capture(self, st: _State, step, iters: int) -> torch.cuda.CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(iters):
                step(st)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step(st)
        return g

    def _spec_pass(self, st: _State, queues, hists) -> int:
        """One verify pass: build draft inputs, run graph, append emitted
        tokens to per-row queues and n-gram histories. Returns mean emit count."""
        B = st.B
        K = R - 1
        rows = []
        for b in range(B):
            draft = hists[b].draft(K) or [queues[b][-1]]
            rows.append([queues[b][-1]] + draft + [0] * (K - len(draft)))
        st.inp_pin.copy_(torch.tensor(rows, dtype=torch.int64))
        st.inp.copy_(st.inp_pin, non_blocking=True)
        if st.graph is not None:
            st.graph.replay()
        else:
            self._decode_step_spec(st)
        st.emit_pin.copy_(st.emit_dev, non_blocking=True)
        torch.cuda.synchronize()
        ep = st.emit_pin
        tot = 0
        for b in range(B):
            m_b = int(ep[b, R].item())
            tot += m_b
            for j in range(m_b):
                t = int(ep[b, j].item())
                queues[b].append(t)
                hists[b].append(t)
        return tot / B

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.dev)
        B, L = ids.shape
        # capacity: prompt + outputs + verify overshoot + warmup/capture writes
        st = self._get_state(B, L + max_new_tokens + 2 * R + 8)
        first = self._prefill(st, ids)
        self._reset_decode(st, L, first)

        if st.graph is None and st.graph1 is None:
            self._self_check(st)
            self._reset_decode(st, L, first)
            try:
                st.graph1 = self._capture(st, self._pick_step(), 2)
                if not self._step_slow_only:
                    try:
                        st.graph = self._capture(st, self._decode_step_spec, 2)
                    except Exception:
                        st.graph = None
            except Exception:
                st.graph1 = False
                if not self._step_slow_only:
                    self._step_slow_only = True
                    self._reset_decode(st, L, first)
                    try:
                        st.graph1 = self._capture(st, self._decode_step_slow, 2)
                    except Exception:
                        st.graph1 = False
            self._reset_decode(st, L, first)

        queues = [[t] for t in first.tolist()]
        hists = [_Ngram(row) for row in input_ids]
        for b in range(B):
            hists[b].append(queues[b][0])

        use_fast = not self._step_slow_only and _HAS_TRITON
        i = 0
        while i < max_new_tokens:
            while min(len(q) for q in queues) <= i:
                if (use_fast and st.graph is not None
                        and st.spec_cooldown == 0):
                    m_mean = self._spec_pass(st, queues, hists)
                    st.spec_window.append(m_mean)
                    if len(st.spec_window) >= 12:
                        # verify pays only if it emits more per pass than it
                        # costs vs a plain decode step; attention work scales
                        # with R, so at big batch*ctx a m<~1.5 mean can lose.
                        if sum(st.spec_window) / len(st.spec_window) < 1.25:
                            st.spec_cooldown = 64
                        st.spec_window.clear()
                else:
                    if st.spec_cooldown:
                        st.spec_cooldown -= 1
                    for b in range(B):
                        st.cur[b, 0] = queues[b][-1]
                    if st.graph1:
                        st.graph1.replay()
                    else:
                        (self._decode_step_fast if use_fast
                         else self._decode_step_slow)(st)
                    st.pin.copy_(st.cur[:, 0], non_blocking=True)
                    torch.cuda.synchronize()
                    for b in range(B):
                        t = int(st.pin[b].item())
                        queues[b].append(t)
                        hists[b].append(t)
            yield [queues[b][i] for b in range(B)]
            i += 1
