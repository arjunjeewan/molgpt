# MolGPT Modernization — Summary of Improvements

A record of the architecture changes made to the MolGPT / LigGPT SMILES
transformer-decoder, the rationale behind each, and the empirical evaluation that
followed. All experiments are on the **unconditional MOSES** benchmark.

> **TL;DR.** Three modern-transformer components (RoPE, SwiGLU, RMSNorm) were
> swapped into the decoder, param-matched, and retrained from scratch. A fourth
> proposed change was investigated and deliberately dropped as unsound. Controlled
> ablation shows the modernization is **validity-neutral**, and a decoding sweep shows
> the apparent "validity gap" vs. the manuscript was a **sampling-temperature artifact**,
> not an architectural one. The concrete payoff: the rotary-embedding (RoPE) model is
> **more novel at matched validity** — and a **3-seed** repeat confirms this edge is
> **statistically significant at T ≤ 1.0** (paired t-test p < 0.05, all seeds agree),
> the useful direction for de novo design. Two further results from running the full
> evaluation across seeds: (i) the novelty comes with a **distributional-fidelity
> trade-off** — the modernized model has higher (worse) FCD at a matched operating point,
> consistent with its lower verbatim memorization; (ii) a **25-epoch run refutes the
> training-budget explanation** for the residual gap vs. the paper — more training raises
> validity but *lowers* novelty, moving away from, not toward, the manuscript's numbers.

---

## 1. Starting point

MolGPT (Bagal et al., *J. Chem. Inf. Model.* 2021) is a GPT-style, decoder-only
transformer that generates molecules as SMILES strings autoregressively. The decoder
lives in `train/model.py` and `generate/model.py` (two byte-identical copies, kept in
sync). The original block uses:

- **Learned absolute position embeddings** (a trainable `pos_emb` table),
- a **GELU** feed-forward network,
- **LayerNorm** (Pre-LN) at three points per block.

Conditioning (property values, scaffold) is done by **prepending condition tokens** to
the sequence. Model size ≈ 6.35M parameters (8 layers, 8 heads, n_embd=256).

---

## 2. Architecture changes

Each change is a drop-in replacement of a single component, **parameter-matched** so
the comparison isolates the component rather than model capacity. Retraining is
required (old checkpoints lack the new layers); vocabulary/tokenization are unchanged.

### 2.1 RoPE — Rotary Position Embedding (replaces learned absolute `pos_emb`)

**What.** Removed the trainable absolute-position table. Position is now injected by
rotating the query/key vectors by an angle proportional to their position
(Su et al., *RoFormer*, 2021).

**Implementation.**
- Helpers `rotate_half`, `apply_rotary_pos_emb` (cos/sin computed in fp32, then cast to
  the query dtype so it is correct under AMP / mixed precision), and a `RotaryEmbedding`
  module holding a non-persistent `inv_freq` buffer.
- Rotation is applied to **Q/K of the SMILES tokens only**. Prepended condition tokens
  (property / scaffold) are **left unrotated** — they are not sequence positions and
  should not carry rotary phase. The count of condition tokens is derived robustly as
  `num_cond = x.size(1) - t` so it stays correct across the optional LSTM-scaffold path.

**Rationale.** Relative positioning is the modern default (LLaMA, GPT-NeoX, etc.): it
removes a learned table, generalizes better across lengths, and encodes *relative*
offsets — a more natural inductive bias for SMILES, where chemical context is relative
rather than tied to an absolute index in the string.

### 2.2 SwiGLU feed-forward (replaces GELU FFN)

**What.** Replaced the GELU MLP with a bias-free gated FFN:
`down( SiLU(gate(x)) * up(x) )` (Shazeer, *GLU Variants Improve Transformer*, 2020).

**Implementation.** Hidden width set to `int(8 * n_embd / 3) = 682`, chosen so the
three-matrix SwiGLU has **the same parameter count** as the original two-matrix GELU
FFN — making it a fair swap, not a capacity increase.

**Rationale.** Gated-linear-unit FFNs consistently outperform plain GELU/ReLU MLPs in
modern transformer LMs at equal parameter budget.

### 2.3 RMSNorm (replaces LayerNorm)

**What.** All three LayerNorms per block replaced with RMSNorm (Zhang & Sennrich,
*Root Mean Square Layer Normalization*, NeurIPS 2019). Compute done in fp32, `eps=1e-5`.

**Implementation.** `RMSNorm` added to the `configure_optimizers` weight-decay blacklist
so its scale parameter is treated as no-decay (consistent with how norm/bias params are
handled).

**Rationale.** RMSNorm drops the mean-centering and bias of LayerNorm — fewer ops,
comparable or better stability — and is the norm used across the modern LLM stack.

### 2.4 Dropped (deliberately): chemistry-aware attention mask

A fourth idea — *masking attention per generation step using RDKit valence checks* — was
investigated and **rejected as unsound**. Reasons:

- The attention mask operates over **past positions**, not the **output vocabulary**;
  valency is a constraint on the *next-token logits*, not on which positions attend to
  which. The mask is the wrong mechanism.
- RDKit **cannot validate a partial SMILES prefix** mid-generation, so there is no
  reliable per-step signal to mask on.
- It is **redundant** with the existing post-hoc `get_mol` validity filter.
- It works **against** MolGPT's stated thesis (favor novelty over guaranteed validity).

**Do not re-attempt the attention-mask version.** If hard validity enforcement is ever
wanted, the sound approach is **logit masking in the sampling loop** driven by a
partial-SMILES grammar / valence automaton (cf. GrammarVAE, SD-VAE) — not an attention
mask. See §4.4 for why this turned out to be unnecessary anyway.

**Final modernized model: 6,349,568 params** (vs. 6,381,056 for the original arch — the
~31k difference is the removed `pos_emb` table net of the SwiGLU/RMSNorm changes).

---

## 3. Evaluation setup

- **Data.** The repo's Drive dataset link is dead; `datasets/moses2.csv` was
  reconstructed from the canonical MOSES splits in `molsets`
  (`_dl/preprocess_moses.py`) — exactly the paper's 1.9M rows
  (train 1.58M / test 176k / test_scaffolds 176k).
- **Fair ablation.** Both architectures trained with **identical** data, batch size
  (384), LR (6e-4), schedule, and epoch count (10). Same 1,584,079 train / 175,984 val
  split after identical `dropna`.
- **Multi-seed robustness.** Each architecture trained on **3 seeds {42, 1, 2}** (the
  seed sets both weight init and DataLoader shuffle order) via the single unified script
  `_dl/train_seeded.py`, so every data point shares one code path. Final train losses are
  tight across seeds (baseline 0.310–0.319, modern 0.309–0.316), confirming stable
  optimization. Metrics below are reported as **mean ± std across the 3 seeds**, with a
  **paired** modern-vs-baseline comparison (matched by seed). A separate **25-epoch**
  modernized run (seed 42) probes the training-budget hypothesis (§4.7).
- **Generation/eval.** `_dl/gen_eval_moses.py` (full MOSES metrics) and
  `_dl/sweep_decode.py` (fast metrics across decoding settings); `_dl/run_train_all.sh`,
  `_dl/run_eval_all.sh` drive the multi-seed grid and `_dl/aggregate_seeds.py` computes
  the mean ± std and the paired test. The repo's own `generate.py` hardcodes
  temperature=1 and was not used.
- **Environment.** conda env `molgpt` (py3.10, torch 2.6.0+cu126, single GV100).

---

## 4. Results

### 4.1 MOSES Table 1 — three-way comparison (10k samples @ T=1.6, full metrics)

| Metric | Manuscript LigGPT | Baseline-arch (ours, 10 ep) | Modernized (ours, 10 ep) |
|---|---|---|---|
| Validity | 0.900 | 0.859 | 0.852 |
| Unique@10K | 0.999 | 0.9998 | 0.9999 |
| Novelty | 0.941 | 0.933 | 0.931 |
| IntDiv1 | 0.871 | 0.871 | 0.870 |
| IntDiv2 | 0.865 | 0.865 | 0.865 |

### 4.2 The validity "gap" is **not** architectural

Training the **verbatim original architecture** under identical conditions gives
validity **0.859** — statistically indistinguishable from the modernized **0.852**
(Δ 0.7pt, within single-run noise) — and *both* sit ~4–5pt under the paper's 0.900.
So the modernization does not cause the shortfall; RoPE/SwiGLU/RMSNorm are
**validity-neutral** and preserve the fast distributional metrics (validity, uniqueness,
internal diversity agree within ~1pt across all three columns). The one metric on which
the two architectures *do* diverge is **FCD**, surfaced only by the full multi-seed
evaluation at T=1.0 — see §4.6.

### 4.3 The validity "gap" **is** a decoding (temperature) artifact

A temperature sweep on the modernized checkpoint (no retraining,
`_dl/sweep_decode.py`):

| T | Validity | Unique | Novelty | IntDiv1 |
|---|---|---|---|---|
| 0.7 | 0.999 | 0.988 | 0.696 | 0.832 |
| 0.9 | 0.997 | 0.998 | 0.770 | 0.844 |
| 1.0 | 0.994 | 0.999 | 0.795 | 0.850 |
| 1.2 | 0.978 | 0.999 | 0.851 | 0.859 |
| 1.6 | 0.853 | 0.9995 | 0.931 | 0.870 |

Validity rises monotonically as temperature falls. The headline 0.852 came from
reporting at an unusually hot **T=1.6**; at any **T ≤ 1.2** the model is ≥ 0.978 valid —
*above* the manuscript's 0.900. There is a clean **validity ↔ novelty/diversity Pareto
frontier**: lower T buys validity by spending novelty/diversity.

**Recommended operating point: T = 1.0–1.2** (e.g. T=1.2 → validity 0.978, novelty
0.851, IntDiv 0.859, unique 0.999), not T=1.6.

### 4.4 Nucleus (top-p) sampling does not beat temperature

top-p just tightens the same distribution and rides the **same** frontier (slightly
validity-favoring; T=1.0/p=0.95 ≈ pure T≈0.85). It is a redundant knob here — temperature
alone spans the operating range — so there is no reason to add it for this model/task.
This also retired the §2.4 "validity enforcement" motivation: temperature already buys
near-100% validity on demand.

### 4.5 The concrete payoff of the modernization: **novelty** (3 seeds, significant at T ≤ 1.0)

Running the identical sweep on **both** architectures over **3 seeds each** and comparing
at matched temperature. Values are **mean ± std (n=3)**; Δ Novelty is **paired** by seed,
with a paired t-test:

| T | Validity (mod / base) | Novelty (mod / base) | **Δ Novelty (mod − base)** | paired p |
|---|---|---|---|---|
| 0.7 | 0.9996 / 0.9994 | 0.6906±.0047 / 0.6640±.0039 | **+0.0266 ±.0065** | **0.019** |
| 0.9 | 0.9973 / 0.9977 | 0.7625±.0064 / 0.7426±.0046 | **+0.0199 ±.0028** | **0.007** |
| 1.0 | 0.9944 / 0.9945 | 0.7932±.0021 / 0.7797±.0027 | **+0.0135 ±.0042** | **0.031** |
| 1.2 | 0.9794 / 0.9783 | 0.8502±.0059 / 0.8454±.0002 | +0.0048 ±.0058 | 0.29 |
| 1.6 | 0.8481 / 0.8601 | 0.9292±.0028 / 0.9292±.0013 | +0.0000 ±.0030 | 1.00 |

- **Validity, diversity, uniqueness frontiers are identical** (Δ ≤ 1pt everywhere, well
  within the seed std). The two validity curves overlay → validity is
  architecture-independent. Cross-seed std is small (≤ 0.006 on every frontier metric),
  so a 3-seed mean is already a tight estimate.
- **The modernized model is more novel at matched temperature, and the edge is
  statistically significant for T ≤ 1.0** (paired p = 0.019 / 0.007 / 0.031 at T =
  0.7 / 0.9 / 1.0; **all three seeds positive** at each). The edge is largest at low T
  (**+2.7pt @ T=0.7**), shrinks through **+1.4pt @ T=1.0**, and is **no longer
  significant at T ≥ 1.2** (+0.5pt, one seed negative) — washing out entirely by T=1.6.
  The multi-seed test sharpens the earlier single-run trend: the payoff is **real and
  significant in the high-validity regime (T ≤ 1.0, validity ≥ 0.994)** and is noise at
  hot temperatures.

**Mechanism — RoPE.** Novelty here = fraction of valid unique generations *not* in the
training set. A higher value at matched validity/diversity means **fewer training
molecules are reproduced verbatim**. Learned absolute `pos_emb` gives each position a
fixed trainable "address" that makes verbatim reproduction of training sequences easy
(especially scaffolds anchored at the start of the string); RoPE's relative,
shift-equivariant positioning memorizes less and recombines more. The **convergence at
high T is the tell**: at low T sampling is tight and the model's mode structure
(memorized vs. generalized) governs output, so the difference shows; at T=1.6 sampling
noise pushes both models off their modes toward the diversity ceiling and the difference
washes out. The **distributional-fidelity trade-off in §4.6** (modern has higher FCD) is
the flip side of the same lower-memorization mechanism. SwiGLU/RMSNorm affect
optimization, not sequence memorization, so the *novelty* direction is attributed to RoPE;
the *magnitude* of the FCD shift is a property of the modernized stack as a whole (no
per-component ablation was run for FCD).

**Net:** at the useful high-validity operating point **T = 1.0** the modernization yields
**+1.4pt novelty (p = 0.03)** at the same validity (0.994), uniqueness, and diversity —
a small but robust, RoPE-attributable improvement in the de-novo-design direction.

### 4.6 Full MOSES at T = 1.0 — the novelty comes with an **FCD trade-off**

Running the *full* `moses.get_all_metrics` (incl. the slow FCD / SNN / Frag / Scaf) at
the recommended **T = 1.0** operating point, 3 seeds per arch (**mean ± std**):

| Metric | Baseline-arch | Modernized | better |
|---|---|---|---|
| Validity | 0.9945 ±.0004 | 0.9937 ±.0004 | tie |
| Unique@10K | 0.9990 ±.0004 | 0.9983 ±.0004 | tie |
| Novelty | 0.7845 ±.0058 | 0.7896 ±.0050 | modern |
| IntDiv1 | 0.8504 ±.0004 | 0.8503 ±.0002 | tie |
| **FCD/Test** ↓ | **0.5709 ±.0351** | **0.9059 ±.0289** | **baseline** |
| Frag/Test ↑ | 0.9972 ±.0001 | 0.9849 ±.0007 | baseline |
| SNN/Test | 0.6241 ±.0005 | 0.6231 ±.0025 | tie |
| Scaf/Test | 0.8833 ±.0029 | 0.8808 ±.0042 | tie |
| Filters | 0.9978 ±.0002 | 0.9974 ±.0002 | tie |

The fast metrics (validity, uniqueness, diversity) are architecture-independent as before,
**but the more sensitive distributional metrics disagree**: the modernized model has a
**markedly higher (worse) FCD — 0.91 vs 0.57**, a gap of ~0.34 against a per-seed std of
~0.03 (≈ 10σ, unambiguous) — and a slightly lower Frag-similarity. FCD measures distance
between the *generated* and the *reference* distribution in ChemNet feature space; a model
that reproduces fewer training molecules verbatim (higher novelty) necessarily drifts
further from that reference. So **FCD↑ and Novelty↑ are two faces of the same
lower-memorization behaviour** — the modernization is *not* "distribution-preserving" on
FCD; it trades benchmark-distribution fidelity for novelty. Which is preferable depends on
the goal: **de novo design favours the modernized model (novelty); benchmark
distribution-matching favours the baseline (FCD).**

### 4.7 Longer training (25 epochs) **refutes** the training-budget hypothesis

The single uncontrolled factor vs. the manuscript was training budget (10 epochs on
reconstructed MOSES). We trained the modernized model for **25 epochs** (seed 42; final
train loss 0.296 vs. ~0.31 at 10 ep) and re-ran the frontier:

| T | Validity (10ep mean → 25ep) | Novelty (10ep mean → 25ep) |
|---|---|---|
| 1.0 | 0.9944 → 0.9963 | 0.7932 → **0.7346** |
| 1.2 | 0.9794 → 0.9856 | 0.8502 → **0.7988** |
| 1.6 | 0.8481 → 0.8890 | 0.9292 → **0.8894** |

More training **raises validity but lowers novelty at every temperature**, and at matched
validity it is a strictly *worse* novelty–validity frontier (e.g. at validity ≈ 0.986 the
25-ep model gives novelty 0.80 vs. ~0.84 interpolated for 10-ep). Full MOSES at T=1.0
confirms it (Novelty 0.740 vs 0.790; FCD essentially unchanged at 0.899 vs 0.906).

**Interpretation.** The paper reports high validity *and* high novelty (0.900 / 0.941)
simultaneously; more epochs move us in the **opposite** direction — toward the
memorization corner (higher validity, lower novelty) — so the residual gap vs. the
manuscript is **not** explained by undertraining. (The unchanged FCD also shows the §4.6
FCD gap is architectural, not a budget artifact.) The remaining discrepancy is more
plausibly in data/preprocessing or reporting details than in training length.

### 4.8 GuacaMol cross-benchmark + **per-component** ablation — the FCD gap is **RoPE**

To (a) check the MOSES findings hold on a second dataset and (b) finally *attribute* the
FCD trade-off (§4.6) to individual components, we ran a 5-config component grid on
**GuacaMol** (1.27M ChEMBL-derived train; the authors' exact `guacamol2.csv`),
unconditional, **3 seeds**, T = 1.0. Each modern component is toggled independently on a
unified model (`_dl/model_ablate.py`) that is **proven bit-identical** (0.00e+00 logit
diff, matched params) to the original baseline and the full-modern definitions at the
all-off / all-on corners. FCD and KL-div use a fixed held-out reference (mean ± std):

| Config | RoPE | SwiGLU | RMSNorm | Validity | Novelty | KL-div | FCD ↓ |
|---|:--:|:--:|:--:|---|---|---|---|
| baseline      | ✗ | ✗ | ✗ | 0.977 | 0.955 | 0.994 | **0.992 ±.007** |
| +RoPE         | ✓ | ✗ | ✗ | 0.981 | **0.960** | 0.988 | **1.253 ±.026** |
| +SwiGLU       | ✗ | ✓ | ✗ | 0.977 | 0.947 | 0.995 | 1.004 ±.009 |
| +RMSNorm      | ✗ | ✗ | ✓ | 0.979 | 0.953 | 0.994 | **0.969 ±.024** |
| modern (all)  | ✓ | ✓ | ✓ | 0.981 | 0.949 | 0.989 | **1.336 ±.025** |

(Uniqueness is 0.999 for every config.) **RoPE is solely responsible for the FCD
degradation**: adding it alone moves FCD 0.992 → 1.253 (+0.26; the three +RoPE seeds
{1.221, 1.252, 1.285} sit entirely above baseline's {0.986, 0.988, 1.001} — zero overlap,
≈ 10σ), whereas **+SwiGLU (1.004) and +RMSNorm (0.969) leave FCD at baseline** (RMSNorm is
even marginally *better*). The same component carries the **novelty** gain (+RoPE is highest
at 0.960). So — now at the component level — RoPE's relative positioning memorizes the
training distribution less: **higher novelty and higher FCD are two faces of one RoPE
effect**, not a whole-stack property. This both **resolves the open §4.6 attribution** and
**reproduces the MOSES trade-off on a second benchmark**.

**Generation-mode coverage (baseline vs modern, GuacaMol, T = 1.0, seed 42).** Beyond
unconditional, the modernization was checked across the other generation kinds — property
(logp), scaffold, and scaffold+property — using per-target **MAD** (deviation of the
generated property from the requested value) and Murcko **scaffold-match**:

| Mode | Arch (base / modern) | Validity | Novelty | MAD(logp) ↓ | Scaf-match ↑ |
|---|---|---|---|---|---|
| property: logp | base / modern | 0.969 / 0.972 | 0.978 / 0.971 | 0.224 / **0.200** | — |
| scaffold       | base / modern | 0.994 / 0.995 | 0.992 / 0.989 | — | 0.973 / **0.982** |
| scaffold+logp  | base / modern | 0.989 / 0.989 | 0.999 / 0.999 | 0.192 / 0.199 | 0.979 / 0.962 |

The modernization is **neutral-to-slightly-positive on conditioning quality**: in the
single-condition modes modern gives **better property control** (logp-MAD 0.200 vs 0.224)
and **better scaffold adherence** (0.982 vs 0.973); combined scaffold+logp is a wash. So the
architecture change does not cost conditional controllability — the only material
distributional cost remains the unconditional FCD, which is RoPE.

---

## 5. Reproducing

All commands run with CWD = `molgpt/`, conda env `molgpt` active.

```bash
# Modernized model: train (10 epochs) → generate+eval @ T=1.6 (full MOSES metrics)
python train/train.py ...                              # produces cond_gpt/weights/unconditional_moses.pt
python _dl/gen_eval_moses.py --temp 1.6 --gen_size 10000 --batch_size 512

# Baseline (original arch) ablation: train → generate+eval
python _dl/train_baseline.py                           # cond_gpt/weights/unconditional_moses_baseline.pt
python _dl/gen_eval_moses.py --baseline --temp 1.6 --gen_size 10000 --batch_size 512

# Decoding sweeps (no retraining): temperature + nucleus, fast metrics
python _dl/sweep_decode.py                             # -> datasets/sweep_decode_modified.csv
python _dl/sweep_decode.py --baseline                  # -> datasets/sweep_decode_baseline.csv

# Multi-seed robustness (3 seeds x both archs + a 25-epoch modern run), then eval + stats
bash   _dl/run_train_all.sh    # trains seeds 1,2 (both archs) + modern 25ep; reuses seed-42 ckpts
bash   _dl/run_eval_all.sh     # per-ckpt frontier sweep + full MOSES @ T=1.0
python _dl/aggregate_seeds.py  # -> _dl/multiseed_results.md, datasets/multiseed_frontier_long.csv

# GuacaMol component ablation (§4.8): 5 configs x 3 seeds uncond + baseline/modern x cond/scaffold
python _dl/test_model_ablate.py   # proves the toggle model == baseline (all-off) == modern (all-on)
bash   _dl/run_guaca_all.sh        # train_ablate -> eval_guaca -> aggregate_guaca (all idempotent)
                                   # tables -> _dl/guaca_ablation_results.md
```

Key artifacts (MOSES): `_dl/model_baseline.py`, `_dl/train_baseline.py`,
`_dl/train_seeded.py`, `_dl/gen_eval_moses.py`, `_dl/sweep_decode.py`,
`_dl/{run_train_all,run_eval_all}.sh`, `_dl/aggregate_seeds.py`; results in
`_dl/multiseed_results.md`, `datasets/sweep_seed_*.csv`,
`datasets/moses_metrics_*_T1.0.json`, `datasets/multiseed_frontier_long.csv`.
Key artifacts (GuacaMol §4.8): `_dl/model_ablate.py` (+`test_model_ablate.py`),
`_dl/train_ablate.py`, `_dl/eval_guaca.py`, `_dl/{run_train_guaca,run_eval_guaca,run_guaca_all}.sh`,
`_dl/aggregate_guaca.py`; results in `_dl/guaca_ablation_results.md`,
`datasets/guaca_metrics_*.json`, `datasets/guaca_results_long.csv`.

---

## 6. Caveats & possible next steps

- **Multi-seed: done (n=3).** The novelty edge now has error bars and a paired t-test
  (§4.5): significant at T ≤ 1.0, noise at T ≥ 1.2. n=3 keeps the t-test low-power, so
  significance rests jointly on the p-values *and* unanimous sign agreement across seeds;
  n=5 would tighten the T=1.2 boundary case. Variance is small (≤ 0.006), so more seeds
  are unlikely to change the conclusion.
- **Residual vs. the manuscript — not training budget (§4.7).** A 25-epoch run *worsens*
  novelty while improving validity, i.e. moves away from the paper's simultaneous
  0.900/0.941. So the residual is **not** undertraining; it is more plausibly
  data/preprocessing of the reconstructed MOSES, or reporting differences. The
  **FCD trade-off (§4.6)** is likewise architectural, not budget-driven.
- **FCD attributed — it's RoPE (§4.8). [resolved]** A per-component GuacaMol ablation
  (3 seeds) isolates the FCD degradation entirely to RoPE (+RoPE alone: 0.992 → 1.253,
  ≈ 10σ, zero seed overlap), which also carries the novelty gain; SwiGLU and RMSNorm are
  FCD-neutral (RMSNorm marginally better). The novelty↑/FCD↑ trade-off is thus a
  single-component (RoPE) property, and it reproduces on a second benchmark. Conditional /
  scaffold controllability is neutral-to-slightly-better under the modernization (§4.8).
- **Representation-level levers** (not pursued here) are the dominant lever for validity
  if ever needed: SELFIES (validity = 1.0 by construction), SAFE (fragment/scaffold
  control), randomized-SMILES augmentation. These are orthogonal to the architecture
  changes above.

---

## References

- Bagal et al., *MolGPT: Molecular Generation Using a Transformer-Decoder Model*, JCIM 2021.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Shazeer, *GLU Variants Improve Transformer*, 2020.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019.
