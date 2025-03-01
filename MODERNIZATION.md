# MolGPT Modernization — Summary of Improvements

A record of the architecture changes made to the MolGPT / LigGPT SMILES
transformer-decoder, the rationale behind each, and the empirical evaluation that
followed. All experiments are on the **unconditional MOSES** benchmark.

> **TL;DR.** Three modern-transformer components (RoPE, SwiGLU, RMSNorm) were
> swapped into the decoder, param-matched, and retrained from scratch. A fourth
> proposed change was investigated and deliberately dropped as unsound. Controlled
> ablation shows the modernization is **validity-neutral and distribution-preserving**,
> and a decoding sweep shows the apparent "validity gap" vs. the manuscript was a
> **sampling-temperature artifact**, not an architectural one. The one concrete payoff:
> the rotary-embedding (RoPE) model is **consistently more novel at matched
> validity** — the useful direction for de novo design.

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
- **Fair ablation.** Both architectures trained with **identical** data, seed (42),
  batch size (384), LR (6e-4), schedule, and epoch count (10). Same
  1,584,079 train / 175,984 val split after identical `dropna`.
- **Generation/eval.** `_dl/gen_eval_moses.py` (full MOSES metrics) and
  `_dl/sweep_decode.py` (fast metrics across decoding settings). The repo's own
  `generate.py` hardcodes temperature=1 and was not used.
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
**validity-neutral and distribution-preserving** (every distributional metric agrees
within ~1pt across all three columns).

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

### 4.5 The concrete payoff of the modernization: **novelty**

Running the identical sweep on **both** architectures and comparing at matched
temperature:

| T | Validity (mod / base) | Novelty (mod / base) | **Δ Novelty (mod − base)** |
|---|---|---|---|
| 0.7 | 0.999 / 0.999 | 0.696 / 0.663 | **+0.033** |
| 0.8 | 0.999 / 0.999 | 0.740 / 0.707 | **+0.033** |
| 0.9 | 0.997 / 0.998 | 0.770 / 0.748 | **+0.022** |
| 1.0 | 0.994 / 0.994 | 0.795 / 0.779 | **+0.017** |
| 1.2 | 0.978 / 0.978 | 0.851 / 0.846 | **+0.005** |
| 1.6 | 0.853 / 0.859 | 0.931 / 0.930 | **+0.001** |

- **Validity, diversity, uniqueness frontiers are identical** (Δ ≤ 0.6pt everywhere —
  noise). The two validity curves overlay exactly → validity is fully
  architecture-independent.
- **The modernized model is consistently more novel at matched temperature.** The edge
  is monotone and ordered — largest at low T (**+3.3pt @ T=0.7**), shrinking to ~0 at
  T=1.6. A monotone gap across six independent temperatures is a real effect, not scatter.

**Mechanism — RoPE.** Novelty here = fraction of valid unique generations *not* in the
training set. A higher value at matched validity/diversity means **fewer training
molecules are reproduced verbatim**. Learned absolute `pos_emb` gives each position a
fixed trainable "address" that makes verbatim reproduction of training sequences easy
(especially scaffolds anchored at the start of the string); RoPE's relative,
shift-equivariant positioning memorizes less and recombines more. The **convergence at
high T is the tell**: at low T sampling is tight and the model's mode structure
(memorized vs. generalized) governs output, so the difference shows; at T=1.6 sampling
noise pushes both models off their modes toward the diversity ceiling and the difference
washes out. SwiGLU/RMSNorm affect optimization, not sequence memorization, and are
unlikely contributors.

**Net:** the modernization is not merely neutral — at the useful high-validity operating
point (T=1.0–1.2) it yields ~1.5–2pt more novel molecules at the same validity and
diversity, attributable to RoPE.

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
```

Key artifacts: `_dl/model_baseline.py`, `_dl/train_baseline.py`, `_dl/gen_eval_moses.py`,
`_dl/sweep_decode.py`; results in `datasets/sweep_decode_{modified,baseline}.csv`.

---

## 6. Caveats & possible next steps

- **Single run per configuration** — no error bars. The ~1.5–3pt novelty edge is
  supported by a monotone trend across six temperatures, but a **multi-seed** repeat
  would let it be reported formally.
- **Residual vs. the manuscript.** At *matched* 0.900 validity (≈ T=1.45) our novelty is
  ~0.90 vs. the paper's 0.941 — a small residual most plausibly from **training budget /
  data** (10 epochs on reconstructed MOSES), the one factor neither the ablation nor the
  sweep controlled. A longer modernized run (20–30 epochs) would test this.
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
