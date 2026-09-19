// Decode-step engine in plain C++ ATen calls.
//
// Identical math to the Python reference step; the only point is skipping
// Python dispatch (~10-15us/op on some runtimes). All tensors are registered
// once via init() and mutated in place, so step() calls carry no marshaling.
#include <torch/extension.h>
#include <vector>

using at::Tensor;

static std::vector<Tensor> W;    // embed, lm, fin; then per-layer: ln_in, wqkv, wo, ln_post, wgu, wd, qkn
static std::vector<Tensor> KC;   // [B, NKV, S, D] per layer
static std::vector<Tensor> VC;
static Tensor CUR, POS, COS_T, SIN_T, SRANGE, INP, EMIT;
static Tensor II, JJ;            // index helpers, [B,1]/[1,NKV] broadcast
static int B_, NL_, NQ_, NKV_, D_, H_, I_, V_, R_, S_;
static double EPS_, SCALE_;

static const double NEG = -std::numeric_limits<double>::infinity();

static inline Tensor rms(const Tensor& x, const Tensor& w) {
    Tensor xf = x.to(at::kFloat);
    Tensor var = xf.pow(2).mean(-1, /*keepdim=*/true);
    return (xf * at::rsqrt(var + EPS_)).to(x.scalar_type()) * w;
}

static inline Tensor rot_half(const Tensor& x) {
    auto xs = x.chunk(2, -1);
    return at::cat({xs[1].neg(), xs[0]}, -1);
}

void init(std::vector<Tensor> weights, std::vector<Tensor> kc,
          std::vector<Tensor> vc, std::vector<Tensor> state,
          int64_t B, int64_t NL, int64_t NQ, int64_t NKV, int64_t D,
          int64_t H, int64_t I, int64_t V, int64_t R, int64_t S,
          double eps, double scale) {
    W = std::move(weights);
    KC = std::move(kc);
    VC = std::move(vc);
    CUR = state[0]; POS = state[1]; COS_T = state[2]; SIN_T = state[3];
    SRANGE = state[4]; INP = state[5]; EMIT = state[6];
    II = state[7]; JJ = state[8];   // raw aranges: [B], [NKV]
    B_ = B; NL_ = NL; NQ_ = NQ; NKV_ = NKV; D_ = D; H_ = H; I_ = I; V_ = V;
    R_ = R; S_ = S; EPS_ = eps; SCALE_ = scale;
}

// Single-token decode: reads CUR/POS, appends K/V, writes next token into CUR
// and bumps POS. Identical op order to Engine._decode_step_slow.
void step() {
    int64_t B = B_;
    Tensor x = at::embedding(W[0], CUR);                      // [B,1,H]
    Tensor cos = COS_T.index_select(0, POS).view({B, 1, 1, D_});
    Tensor sin = SIN_T.index_select(0, POS).view({B, 1, 1, D_});
    Tensor nvalid = SRANGE.unsqueeze(0) > POS.unsqueeze(1);   // [B,S] not-valid
    Tensor pi = POS.view({B, 1}).expand({B, NKV_});
    int64_t qkd = (NQ_ + NKV_) * D_;
    for (int64_t i = 0; i < NL_; ++i) {
        int64_t b = 3 + i * 7;
        Tensor h = rms(x, W[b + 0]).view({B, H_});
        Tensor qkv = h.matmul(W[b + 1].t());
        Tensor qk = qkv.narrow(1, 0, qkd).view({B, 1, NQ_ + NKV_, D_});
        Tensor v = qkv.narrow(1, qkd, NKV_ * D_).view({B, 1, NKV_, D_});
        Tensor qkn = rms(qk, W[b + 6]);
        Tensor qke = qkn * cos + rot_half(qkn) * sin;
        Tensor qe = qke.narrow(2, 0, NQ_);
        Tensor ke = qke.narrow(2, NQ_, NKV_);
        Tensor bi = II.view({B, 1}).expand({B, NKV_});
        Tensor gj = JJ.view({1, NKV_}).expand({B, NKV_});
        KC[i].index_put_({bi, gj, pi}, ke.squeeze(1));
        VC[i].index_put_({bi, gj, pi}, v.squeeze(1));
        Tensor qg = qe.reshape({B, NKV_, NQ_ / NKV_, D_});
        Tensor scores = qg.matmul(KC[i].transpose(-1, -2)) * SCALE_;
        scores.masked_fill_(nvalid.unsqueeze(1).unsqueeze(1), NEG);
        Tensor p = at::softmax(scores.to(at::kFloat), -1).to(at::kBFloat16);
        Tensor o = p.matmul(VC[i]).reshape({B, NQ_ * D_});
        x = at::addmm(x.view({B, H_}), o, W[b + 2].t()).view({B, 1, H_});
        Tensor h2 = rms(x, W[b + 3]).view({B, H_});
        Tensor gu = h2.matmul(W[b + 4].t());
        Tensor m = at::silu(gu.narrow(1, 0, I_)) * gu.narrow(1, I_, I_);
        x = at::addmm(x.view({B, H_}), m, W[b + 5].t()).view({B, 1, H_});
    }
    x = rms(x, W[2]);
    Tensor logits = x.view({B, H_}).matmul(W[1].t());
    Tensor tok = logits.argmax(-1);
    CUR.copy_(tok.view({B, 1}));
    POS.add_(1);
}

// R-row verify pass: reads INP [B,R] and POS, appends R keys/values per row,
// writes [B,R] argmax grid + accept count m into EMIT [B,R+1], POS += m.
// Identical op order to Engine._decode_step_slow_batch.
void step_batch() {
    int64_t B = B_, R = R_;
    auto opt = INP.options();
    Tensor rr = at::arange(R, opt.dtype(at::kLong));
    Tensor pos_r = POS.unsqueeze(1) + rr.unsqueeze(0);        // [B,R]
    Tensor x = at::embedding(W[0], INP);                      // [B,R,H]
    Tensor pos_flat = pos_r.reshape({-1});
    Tensor cos = COS_T.index_select(0, pos_flat).view({B, R, 1, D_});
    Tensor sin = SIN_T.index_select(0, pos_flat).view({B, R, 1, D_});
    Tensor nvalid = SRANGE.unsqueeze(0).unsqueeze(0) >
                    pos_r.unsqueeze(-1);                      // [B,R,S] not-valid
    Tensor bi = II.view({B, 1, 1}).expand({B, R, NKV_});
    Tensor gi = JJ.view({1, 1, NKV_}).expand({B, R, NKV_});
    Tensor pi = pos_r.unsqueeze(-1).expand({B, R, NKV_});
    int64_t qkd = (NQ_ + NKV_) * D_;
    for (int64_t i = 0; i < NL_; ++i) {
        int64_t b = 3 + i * 7;
        Tensor h = rms(x, W[b + 0]).view({B * R, H_});
        Tensor qkv = h.matmul(W[b + 1].t());
        Tensor qk = qkv.narrow(1, 0, qkd).view({B, R, NQ_ + NKV_, D_});
        Tensor v = qkv.narrow(1, qkd, NKV_ * D_).view({B, R, NKV_, D_});
        Tensor qkn = rms(qk, W[b + 6]);
        Tensor qke = qkn * cos + rot_half(qkn) * sin;
        Tensor qe = qke.narrow(2, 0, NQ_);
        Tensor ke = qke.narrow(2, NQ_, NKV_);
        KC[i].index_put_({bi, gi, pi}, ke);
        VC[i].index_put_({bi, gi, pi}, v);
        Tensor qg = qe.reshape({B, R, NKV_, NQ_ / NKV_, D_});
        Tensor scores = qg.matmul(
            KC[i].unsqueeze(1).transpose(-1, -2)) * SCALE_;   // [B,R,NKV,G,S]
        scores.masked_fill_(nvalid.unsqueeze(2).unsqueeze(2), NEG);
        Tensor p = at::softmax(scores.to(at::kFloat), -1).to(at::kBFloat16);
        Tensor o = p.matmul(
            VC[i].unsqueeze(1).expand({B, R, NKV_, S_, D_})); // [B,R,NKV,G,D]
        x = at::addmm(x.view({B * R, H_}), o.reshape({B * R, NQ_ * D_}),
                      W[b + 2].t()).view({B, R, H_});
        Tensor h2 = rms(x, W[b + 3]).view({B * R, H_});
        Tensor gu = h2.matmul(W[b + 4].t());
        Tensor m = at::silu(gu.narrow(1, 0, I_)) * gu.narrow(1, I_, I_);
        x = at::addmm(x.view({B * R, H_}), m, W[b + 5].t()).view({B, R, H_});
    }
    x = rms(x, W[2]);
    Tensor logits = x.view({B * R, H_}).matmul(W[1].t());
    Tensor am = logits.view({B, R, V_}).argmax(-1);           // [B,R]
    Tensor matched = (am.narrow(1, 0, R - 1) ==
                      INP.narrow(1, 1, R - 1)).to(at::kLong);
    Tensor m = matched.cumprod(1).sum(1) + 1;                 // [B]
    POS.add_(m);
    EMIT.copy_(at::cat({am, m.view({B, 1})}, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "register state and weights");
    m.def("step", &step, "single-token decode");
    m.def("step_batch", &step_batch, "R-row verify pass");
}
